# Copyright 2026 The LiteRT CLI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generic PyTorch script conversion logic for LiteRT CLI."""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys
from typing import Any
import uuid

import click
from litert_cli.core import npu_utils

from ai_edge_litert.aot.ai_pack import export_lib as ai_pack_export

# Generative quantization recipes for script mode: name -> (mode, weight
# dtype) passed to litert_torch's quant_recipe_utils.
_GENERATIVE_RECIPES = {
    "dynamic_wi8_afp32": ("dynamic", "INT8"),
    "dynamic_wi4_afp32": ("dynamic", "INT4"),
    "weight_only_wi8_afp32": ("weight_only", "INT8"),
    "weight_only_wi4_afp32": ("weight_only", "INT4"),
    "fp16": ("fp16", None),
}
# PT2E recipes: name -> (is_dynamic, is_per_channel).
_PT2E_RECIPES = {
    "pt2e_static": (False, False),
    "pt2e_static_per_channel": (False, True),
    "pt2e_dynamic": (True, False),
    "pt2e_dynamic_per_channel": (True, True),
}


def parse_model_args(model_args: str | None) -> dict[str, Any]:
  """Parses `--model-args` ("k=v,k2=v2") into keyword arguments.

  Values are Python literals when they parse as one (ints, floats, bools,
  None, quoted strings), and plain strings otherwise.

  Args:
    model_args: The option value.

  Raises:
    click.BadParameter: If an item is not of the form key=value.
  """
  parsed_args = {}
  if not model_args:
    return parsed_args
  for item in model_args.split(","):
    key, sep, value = item.partition("=")
    key = key.strip()
    if not sep or not key:
      raise click.BadParameter(
          f"'{item}' is not of the form key=value.", param_hint="--model-args"
      )
    value = value.strip()
    if value.lower() in ("true", "false"):
      parsed_args[key] = value.lower() == "true"
      continue
    try:
      parsed_args[key] = ast.literal_eval(value)
    except (ValueError, SyntaxError):
      parsed_args[key] = value
  return parsed_args


def _quant_config_for_recipe(quantize: str) -> Any:
  """Returns the litert_torch QuantConfig for a script-mode recipe name.

  Args:
    quantize: The --quantize value; 'none' means no quantization.

  Raises:
    click.BadParameter: If the recipe is unknown.
  """
  name = quantize.strip().lower()
  if name == "none":
    return None
  if name not in _GENERATIVE_RECIPES and name not in _PT2E_RECIPES:
    raise click.BadParameter(
        f"Unknown recipe '{quantize}' for a PyTorch script. Supported:"
        f" {', '.join(sorted([*_GENERATIVE_RECIPES, *_PT2E_RECIPES, 'none']))}.",
        param_hint="--quantize",
    )
  # pylint: disable=g-import-not-at-top
  from litert_torch.quantize import quant_config as quant_config_lib

  click.echo(f"Building QuantConfig for recipe '{name}'...")
  if name in _PT2E_RECIPES:
    from litert_torch.quantize import pt2e_quantizer as pt2eq

    is_dynamic, is_per_channel = _PT2E_RECIPES[name]
    pt2e_cfg = pt2eq.get_symmetric_quantization_config(
        is_per_channel=is_per_channel, is_dynamic=is_dynamic
    )
    q = pt2eq.PT2EQuantizer().set_global(pt2e_cfg)
    return quant_config_lib.QuantConfig(pt2e_quantizer=q)
  from litert_torch.generative.quantize import quant_attrs
  from litert_torch.generative.quantize import quant_recipe
  from litert_torch.generative.quantize import quant_recipe_utils

  mode, dtype = _GENERATIVE_RECIPES[name]
  if mode == "fp16":
    recipe_obj = quant_recipe_utils.create_layer_quant_fp16()
  elif mode == "weight_only":
    recipe_obj = quant_recipe_utils.create_layer_quant_weight_only(
        getattr(quant_attrs.Dtype, dtype)
    )
  else:
    recipe_obj = quant_recipe_utils.create_layer_quant_dynamic(
        getattr(quant_attrs.Dtype, dtype)
    )
  return quant_config_lib.QuantConfig(
      generative_recipe=quant_recipe.GenerativeQuantRecipe(default=recipe_obj)
  )


def convert_generic_script(
    script: str,
    model_func: str,
    input_func: str,
    output: str,
    target: tuple[str, ...],
    export_aipack: pathlib.Path | None,
    quantize: str | None = None,
    model_args: str | None = None,
) -> None:
  """Converts using generic PyTorch scripts and `litert_torch.convert`.

  Args:
    script: Path to the PyTorch script (.py).
    model_func: Name of function returning the `torch.nn.Module`.
    input_func: Name of function returning sample inputs.
    output: Directory to save the converted model.
    target: NPU targets to apply AOT compilation.
    export_aipack: Output directory to export the AI Pack for PODAI.
    quantize: Quantization recipe to apply.
    model_args: Arguments to pass to custom model/input functions.

  Raises:
    ImportError: If the script loading fails.
    AttributeError: If functions are missing in the user script.
    ValueError: If inputs result shape is not supported.
  """
  # pylint: disable=g-import-not-at-top
  import litert_torch

  if target and export_aipack:
    npu_utils.check_ai_pack_dir(export_aipack)

  script_path = pathlib.Path(script).resolve()
  click.echo(f"Loading custom script from: {script_path}")

  # Dynamically load the user's python file with sandbox isolation
  module_name = f"user_model_script_{uuid.uuid4().hex}"
  spec = importlib.util.spec_from_file_location(module_name, script_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load script {script_path}")

  user_module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = user_module

  # Add the script's directory to sys.path so it can resolve its own local imports
  sys.path.insert(0, str(script_path.parent))

  try:
    spec.loader.exec_module(user_module)

    # Look up the model factory
    if not hasattr(user_module, model_func):
      raise AttributeError(
          f"Function '{model_func}' not found in {script_path}"
      )
    # Look up the args factory
    if not hasattr(user_module, input_func):
      raise AttributeError(
          f"Function '{input_func}' not found in {script_path}"
      )

    parsed_args = parse_model_args(model_args)

    click.echo(
        f"Instantiating model via '{model_func}' with args: {parsed_args}..."
    )
    model_result = getattr(user_module, model_func)(**parsed_args)

    # The user might return just the nn.Module or a tuple of (nn.Module, QuantConfig)
    if isinstance(model_result, tuple):
      model, quant_config = model_result
    else:
      model = model_result
      quant_config = None

    if quant_config is None and quantize:
      quant_config = _quant_config_for_recipe(quantize)

    click.echo(
        f"Generating sample inputs via '{input_func}' with args:"
        f" {parsed_args}..."
    )
    inputs_result = getattr(user_module, input_func)(**parsed_args)

    sample_args: tuple[Any, ...]
    sample_kwargs: dict[str, Any]
    if isinstance(inputs_result, tuple):
      sample_args = inputs_result
      sample_kwargs = {}
    elif isinstance(inputs_result, dict):
      sample_args = ()
      sample_kwargs = inputs_result
    else:
      raise ValueError(
          f"'{input_func}' must return a tuple (args) or a dict (kwargs)."
      )

    click.echo("Executing liteRT conversion tracer...")

    builder = litert_torch
    if target:
      for target_obj in npu_utils.get_aot_targets(
          target, export_aipack=bool(export_aipack)
      ):
        builder = builder.experimental_add_compilation_backend(target_obj)

    try:
      edge_model = builder.convert(
          module=model,
          sample_args=sample_args,
          sample_kwargs=sample_kwargs,
          quant_config=quant_config,
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      hint = npu_utils.compile_failure_hint(e) if target else None
      if hint is None:
        raise
      raise click.ClickException(
          f"AOT compilation for {', '.join(target)} failed: {e!r}\n{hint}"
      ) from e

    out_path = pathlib.Path(output)
    out_path.mkdir(parents=True, exist_ok=True)
    model_name = script_path.stem

    if target and export_aipack:
      export_dir = pathlib.Path(export_aipack)
      click.echo(f"Exporting AI Pack to: {export_dir}")
      export_dir.mkdir(parents=True, exist_ok=True)
      ai_pack_export.export(edge_model, str(export_dir), model_name, "model")
    else:
      if target:
        click.echo(f"Exporting compiled model to {out_path} as {model_name}...")
        edge_model.export(str(out_path), model_name=model_name)
      else:
        final_path = out_path / f"{model_name}.tflite"
        click.echo(f"Exporting converted model to {final_path}...")
        edge_model.export(str(final_path))

    click.echo("Done!")

  finally:
    # Cleanup injected sys.path modification and sandbox module
    sys.modules.pop(module_name, None)
    if sys.path[0] == str(script_path.parent):
      sys.path.pop(0)
