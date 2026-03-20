"""Quantize a LiteRT model using AI Edge Quantizer."""

from __future__ import annotations

import importlib.util
import pathlib

import click
from litert_cli.core import deps


@click.command(
    "quantize",
    help="""Quantize a LiteRT model.

MODEL: Path to the input .tflite model.

Examples:

  Dynamic INT8 Quantization (Default):

    $ litert quantize raw_model.tflite --output quant_model.tflite

  Weight-Only INT8 Quantization:

    $ litert quantize raw_model.tflite --output quant_model.tflite \
        --type int8_weight_only

  Static Quantization (Requires calibration data):

    $ litert quantize raw_model.tflite --type static \
        --calibration-data calib_data.py --output quant_model.tflite

  Custom Recipe:

    $ litert quantize raw_model.tflite --recipe recipe.json \
        --output quant_model.tflite
""",
)
@click.argument(
    "model",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, writable=True, path_type=pathlib.Path),
    required=False,
    default=None,
    help="Path to save the output quantized .tflite model.",
)
@click.option(
    "--type",
    "quant_type",
    type=click.Choice(
        ["int8_dynamic", "int8_weight_only", "int16_weight_only", "static"]
    ),
    default="int8_dynamic",
    help="Type of quantization to apply.",
)
@click.option(
    "--calibration-data",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
    help=(
        "Path to Python script providing calibration data (required for"
        " 'static')."
    ),
)
@click.option(
    "--recipe",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
    help="Path to JSON recipe file for custom configurations.",
)
@deps.require_extra("quantize")
def quantize_cmd(
    model: pathlib.Path,
    output: pathlib.Path | None,
    quant_type: str,
    calibration_data: pathlib.Path | None,
    recipe: pathlib.Path | None,
) -> None:
  r"""Quantizes a LiteRT model.

  Args:
    model: Path to the input .tflite model.
    output: Path to save the output quantized .tflite model.
    quant_type: Type of quantization to apply.
    calibration_data: Path to Python script providing calibration data.
    recipe: Path to JSON recipe file for custom configurations.
  """
  import ai_edge_quantizer as aeq  # pylint: disable=g-import-not-at-top

  if not output:
    output = model.with_name(f"{model.stem}_quant.tflite")

  click.echo(f"Quantizing '{model}' to '{output}'...")
  quantizer = aeq.Quantizer(str(model))

  if recipe:
    click.echo(f"Loading recipe from '{recipe}'...")
    quantizer.load_quantization_recipe(str(recipe))
  elif quant_type == "static":
    click.echo("Configuring static quantization (A8W8)...")
    quantizer.add_static_config(
        regex=".*",
        operation_name=aeq.qtyping.TFLOperationName.ALL_SUPPORTED,
        activation_num_bits=8,
        weight_num_bits=8,
    )
  elif quant_type == "int8_dynamic":
    quantizer.add_dynamic_config(
        regex=".*",
        operation_name=aeq.qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=8,
    )
  elif quant_type == "int8_weight_only":
    quantizer.add_weight_only_config(
        regex=".*",
        operation_name=aeq.qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=8,
    )
  elif quant_type == "int16_weight_only":
    quantizer.add_weight_only_config(
        regex=".*",
        operation_name=aeq.qtyping.TFLOperationName.ALL_SUPPORTED,
        num_bits=16,
    )

  calibration_result = None
  if quantizer.need_calibration:
    if not calibration_data:
      raise click.UsageError(
          "Calibration data is required for this configuration (Static or"
          " specific recipes). Use --calibration-data <script.py>"
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
  output.parent.mkdir(parents=True, exist_ok=True)
  result.export_model(str(output), overwrite=True)
  click.secho(f"Quantization complete! Saved to {output}", fg="green")
