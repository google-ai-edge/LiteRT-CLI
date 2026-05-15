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

"""Quantize a LiteRT model using AI Edge Quantizer."""

from __future__ import annotations

import importlib.util
import pathlib
import textwrap

import click
from litert_cli.core import deps


@click.command(
    "quantize",
    help=textwrap.dedent("""\
        Quantize a LiteRT model.

        MODEL: Path to the input .tflite model.

        Examples:

          Dynamic INT8 Quantization (Default):

            $ litert quantize raw_model.tflite --output quant_model.tflite

          Weight-Only INT8 Quantization:

            $ litert quantize raw_model.tflite --output quant_model.tflite \
                --type weight_only_wi8_afp32

          Static Quantization (Requires calibration data):

            $ litert quantize raw_model.tflite --recipe static_wi8_ai8 \
                --calibration-data calib_data.py --output quant_model.tflite

          Custom Recipe:

            $ litert quantize raw_model.tflite --custom-recipe recipe.json \
                --output quant_model.tflite
        """),
)
@click.argument("model", type=str)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
    required=False,
    default=None,
    help="Path to save the output quantized .tflite model.",
)
@click.option(
    "--recipe",
    "quant_recipe",
    type=str,
    default="dynamic_wi8_afp32",
    help=(
        "Built-in quantization recipe to apply (e.g., dynamic_wi8_afp32,"
        " weight_only_wi8_afp32, static_wi8_ai8). See 'ai_edge_quantizer.recipe'"
        " for full list."
    ),
)
@click.option(
    "--calibration-data",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
    help=(
        "Path to Python script providing calibration data (required for"
        " 'static_wi8_ai8')."
    ),
)
@click.option(
    "--custom-recipe",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
    help="Path to JSON recipe file for custom configurations.",
)
@deps.require_extra("quantize")
def quantize_cmd(
    model: str,
    output: pathlib.Path | None,
    quant_recipe: str,
    calibration_data: pathlib.Path | None,
    custom_recipe: pathlib.Path | None,
) -> None:
  r"""Quantize a LiteRT model.

  Args:
    model: Path to the input .tflite model or Model Reference.
    output: Path to save the output quantized .tflite model.
    quant_recipe: Built-in quantization recipe to apply.
    calibration_data: Path to Python script providing calibration data.
    custom_recipe: Path to JSON recipe file for custom configurations.

  Raises:
    click.UsageError: If calibration data is missing when required.
    click.ClickException: If calibration script loading fails or is invalid.
  """
  from ai_edge_quantizer import quantizer as aeq  # pylint: disable=g-import-not-at-top
  from ai_edge_quantizer import qtyping  # pylint: disable=g-import-not-at-top

  from litert_cli.core import models as core_models

  resolved_model_path, _ = core_models.resolve_model_reference(model)

  if str(resolved_model_path) != str(model):
    click.echo(f"Resolved model '{model}' to '{resolved_model_path}'")

  model_path = pathlib.Path(resolved_model_path)

  if quant_recipe == "static_wi8_ai8" and calibration_data is None:
    raise click.UsageError(
        "--calibration-data is required when --recipe is 'static_wi8_ai8'."
    )

  resolved_output = output or model_path.with_name(
      f"{model_path.stem}_quant.tflite"
  )

  click.echo(f"Quantizing '{model_path}' to '{resolved_output}'...")
  quantizer = aeq.Quantizer(str(model_path))

  if custom_recipe:
    click.echo(f"Loading custom recipe from '{custom_recipe}'...")
    import json

    with open(custom_recipe, "r") as f:
      recipe_content = json.load(f)
    quantizer.load_quantization_recipe(recipe_content)
  else:
    from ai_edge_quantizer import recipe as aeq_recipe

    if hasattr(aeq_recipe, quant_recipe):
      click.echo(f"Loading built-in recipe '{quant_recipe}'...")
      recipe_obj = getattr(aeq_recipe, quant_recipe)()
      quantizer.load_quantization_recipe(recipe_obj)
    else:
      click.echo(f"Fallback configuring dynamic quantization for '{quant_recipe}'...")
      quantizer.add_dynamic_config(
          regex=".*",
          operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
          num_bits=8,
      )

  calibration_result = None
  if quantizer.need_calibration:
    if calibration_data is None:
      raise click.UsageError(
          "Calibration data is required for the specified recipe. Use "
          "--calibration-data <script.py>"
      )
    click.echo(f"Loading calibration data from '{calibration_data}'...")

    spec = importlib.util.spec_from_file_location(
        "calib_module", str(calibration_data)
    )
    if spec is None or spec.loader is None:
      raise click.ClickException(
          f"Failed to load calibration script from '{calibration_data}'"
      )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_calibration_data"):
      raise click.ClickException(
          "Calibration script must define 'get_calibration_data()' function."
      )
    calib_data = module.get_calibration_data()
    click.echo("Running calibration...")
    calibration_result = quantizer.calibrate(calib_data)

  click.echo("Applying quantization...")
  result = quantizer.quantize(calibration_result)
  resolved_output.parent.mkdir(parents=True, exist_ok=True)
  result.export_model(str(resolved_output), overwrite=True)

  click.secho(f"Quantization complete! Saved to {resolved_output}", fg="green")
