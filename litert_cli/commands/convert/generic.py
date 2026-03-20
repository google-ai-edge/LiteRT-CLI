"""Generic PyTorch script conversion logic for LiteRT CLI."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import click


def convert_generic_script(
    script: str, model_func: str, input_func: str, output: str
) -> None:
  """Conversion logic using generic PyTorch scripts and `litert_torch.convert`.

  Args:
    script: Path to the PyTorch script (.py).
    model_func: Name of function returning the `torch.nn.Module`.
    input_func: Name of function returning sample inputs.
    output: Directory to save the converted model.

  Raises:
    ImportError: If the script loading fails.
    AttributeError: If functions are missing in the user script.
    ValueError: If inputs result shape is not supported.
  """
  # pylint: disable=g-import-not-at-top
  import litert_torch

  script_path = pathlib.Path(script).resolve()
  click.echo(f"Loading custom script from: {script_path}")

  # Dynamically load the user's python file
  module_name = "user_model_script"
  spec = importlib.util.spec_from_file_location(module_name, script_path)
  if spec is None or spec.loader is None:
    raise ImportError(f"Could not load script {script_path}")

  user_module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = user_module

  # Add the script's directory to sys.path so it can resolve its own local
  # imports
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

    click.echo(f"Instantiating model via '{model_func}'...")
    model_result = getattr(user_module, model_func)()

    # The user might return just the nn.Module or a tuple of
    # (nn.Module, QuantConfig)
    if isinstance(model_result, tuple):
      model = model_result[0]
      quant_config = model_result[1]
    else:
      model = model_result
      quant_config = None

    click.echo(f"Generating sample inputs via '{input_func}'...")
    inputs_result = getattr(user_module, input_func)()

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
    # Delegate to the core litert_torch tracer
    edge_model = litert_torch.convert(
        module=model,
        sample_args=sample_args,
        sample_kwargs=sample_kwargs,
        quant_config=quant_config,
    )

    out_path = pathlib.Path(output)
    out_path.mkdir(parents=True, exist_ok=True)
    model_name = script_path.stem

    final_path = out_path / f"{model_name}.tflite"
    click.echo(f"Exporting converted model to {final_path}...")
    edge_model.export(str(final_path))
    click.echo("Done!")

  finally:
    # Cleanup injected sys.path modification
    if sys.path[0] == str(script_path.parent):
      sys.path.pop(0)
