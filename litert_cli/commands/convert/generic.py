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

import importlib.util
import pathlib
import shutil
import sys
from typing import Any
import uuid

import click
from litert_cli.core import npu_utils

from ai_edge_litert.aot.ai_pack import export_lib as ai_pack_export


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

    # Parse model_args
    parsed_args = {}
    if model_args:
      for item in model_args.split(","):
        if "=" in item:
          k, v = item.split("=", 1)
          if v.isdigit():
            parsed_args[k] = int(v)
          elif v.replace(".", "", 1).isdigit():
            parsed_args[k] = float(v)
          elif v.lower() == "true":
            parsed_args[k] = True
          elif v.lower() == "false":
            parsed_args[k] = False
          else:
            parsed_args[k] = v

    click.echo(f"Instantiating model via '{model_func}' with args: {parsed_args}...")
    model_result = getattr(user_module, model_func)(**parsed_args)

    # The user might return just the nn.Module or a tuple of (nn.Module, QuantConfig)
    if isinstance(model_result, tuple):
      model, quant_config = model_result
    else:
      model = model_result
      quant_config = None

    if quant_config is None and quantize:
      click.echo(f"Building dynamic QuantConfig for recipe '{quantize}'...")
      from litert_torch.quantize import quant_config as quant_config_lib

      if "pt2e" in quantize.lower():
        from litert_torch.quantize import pt2e_quantizer as pt2eq

        is_dynamic = "dynamic" in quantize.lower()
        is_per_channel = "per_channel" in quantize.lower()
        pt2e_cfg = pt2eq.get_symmetric_quantization_config(
            is_per_channel=is_per_channel, is_dynamic=is_dynamic
        )
        q = pt2eq.PT2EQuantizer().set_global(pt2e_cfg)
        quant_config = quant_config_lib.QuantConfig(pt2e_quantizer=q)
      else:
        from ai_edge_quantizer import recipe as recipe_lib
        from litert_torch.generative.quantize import quant_recipe

        if hasattr(recipe_lib, quantize):
          recipe_obj = getattr(recipe_lib, quantize)()
        elif hasattr(recipe_lib, quantize + "_recipe"):
          recipe_obj = getattr(recipe_lib, quantize + "_recipe")()
        else:
          recipe_obj = recipe_lib.dynamic_wi8_afp32()
        quant_config = quant_config_lib.QuantConfig(
            generative_recipe=quant_recipe.GenerativeQuantRecipe(
                default=recipe_obj
            )
        )

    click.echo(f"Generating sample inputs via '{input_func}' with args: {parsed_args}...")
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
      for t in target:
        target_obj = npu_utils.get_target(t)
        builder = builder.experimental_add_compilation_backend(target_obj)

    edge_model = builder.convert(
        module=model,
        sample_args=sample_args,
        sample_kwargs=sample_kwargs,
        quant_config=quant_config,
    )

    out_path = pathlib.Path(output)
    out_path.mkdir(parents=True, exist_ok=True)
    model_name = script_path.stem

    if target and export_aipack:
      export_dir = pathlib.Path(export_aipack)
      click.echo(f"Exporting AI Pack to: {export_dir}")
      shutil.rmtree(export_dir, ignore_errors=True)
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
