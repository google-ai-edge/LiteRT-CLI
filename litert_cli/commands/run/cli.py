"""Command Line Interface for executing LiteRT models.

This module provides the `litert run` command, which allows users to
execute TFLite models on either the local desktop or a connected Android device.

Key Features:
- Desktop Execution: Uses the LiteRT Python API (`CompiledModel`) to run
  inference locally. It automatically inspects the model signature,
  generates appropriate dummy input data, and prints the output tensor
  statistics.
- Android Execution: Seamlessly pushes the model and the compiled `run_model`
  binary to an attached Android device via `adb`, and executes it remotely.
"""

from __future__ import annotations

import pathlib

import click
from litert_cli.core import deps


@click.command(
    "run",
    help="""Run LiteRT models locally or on device.

MODEL: Path to the LiteRT model (.tflite).

Examples:

  1. Run on desktop (CPU) with dummy inputs:

    $ litert run model.tflite

  2. Run on desktop with GPU acceleration:

    $ litert run model.tflite --gpu

  3. Run with custom inputs (path or literal):

    $ litert run model.tflite --input image.jpg

    OR with named inputs:

    $ litert run model.tflite --input in1=1.0 --input in2=image.jpg

  4. Run on an attached Android device:

    $ litert run model.tflite --android

  5. Run on Android with GPU acceleration:

    $ litert run model.tflite --android --gpu

  6. Benchmark execution with 10 iterations:

    $ litert run model.tflite --iterations 10

  7. Print detailed tensor outputs:

    $ litert run model.tflite --print_tensors --sample_size 10
""",
)
@deps.require_extra("run")
@click.argument(
    "model",
    type=click.Path(
        exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--input",
    "inputs",
    multiple=True,
    help=(
        "Input data for the model. Can be a literal array (e.g. '[1,2]'), "
        "a path to an image/npy/raw file. "
        "You can specify multiple inputs using format: --input name=value "
        "or just --input value if the model has only one input."
    ),
)
@click.option(
    "--desktop",
    "target",
    flag_value="desktop",
    default=True,
    help="Target desktop platform to run (Default).",
)
@click.option(
    "--android",
    "target",
    flag_value="android",
    help="Target Android platform to run.",
)
@click.option(
    "--cpu",
    "accelerator",
    flag_value="cpu",
    default=True,
    help="Use CPU accelerator (Default).",
)
@click.option(
    "--gpu",
    "accelerator",
    flag_value="gpu",
    help="Use GPU accelerator.",
)
@click.option(
    "--npu",
    "accelerator",
    flag_value="npu",
    help="Use NPU accelerator.",
)
@click.option(
    "--signature_index",
    type=int,
    default=0,
    help="Index of model signature to run. Default is 0.",
)
@click.option(
    "--iterations",
    type=int,
    default=1,
    help="Number of times to execute the model for benchmarking. Default is 1.",
)
@click.option(
    "--print_tensors",
    is_flag=True,
    default=False,
    help="Print output tensor values after execution.",
)
@click.option(
    "--sample_size",
    type=int,
    default=5,
    help="Number of sample elements to print from tensors. Default is 5.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Silence C++ INFO and WARNING logs during execution.",
)
def run_cmd(
    model: pathlib.Path,
    inputs: tuple[str, ...],
    target: str,
    accelerator: str,
    signature_index: int,
    iterations: int,
    print_tensors: bool,
    sample_size: int,
    quiet: bool,
) -> None:
  r"""Runs LiteRT models locally or on device.

  Args:
    model: Path to the LiteRT model (.tflite).
    inputs: Tuple of input assignments (e.g., 'name=value' or just 'value').
    target: Execution target ('desktop' or 'android').
    accelerator: Hardware accelerator ('cpu', 'gpu', or 'npu').
    signature_index: Signature index to invoke.
    iterations: Number of iterations to run the model.
    print_tensors: Whether to print output tensor elements.
    sample_size: Number of elements to print per tensor.
  """
  if target == "desktop":
    from litert_cli.commands.run import desktop  # pylint: disable=g-import-not-at-top

    desktop.run_desktop(
        str(model),
        inputs=inputs,
        accelerator=accelerator,
        signature_index=signature_index,
        iterations=iterations,
        print_tensors=print_tensors,
        sample_size=sample_size,
        quiet=quiet,
    )
  elif target == "android":
    from litert_cli.commands.run import android  # pylint: disable=g-import-not-at-top

    android.run_android(
        str(model),
        inputs=inputs,
        accelerator=accelerator,
        signature_index=signature_index,
        iterations=iterations,
        print_tensors=print_tensors,
        sample_size=sample_size,
    )
  else:
    click.secho(f"Target '{target}' is not yet supported.", fg="red")
