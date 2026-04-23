"""CLI Benchmark Module providing commands to benchmark LiteRT models.

This module defines the `benchmark` command using Click. It supports:
1. Benchmarking on a local Android device via adb (Default).
2. Benchmarking on Google Cloud Platform (GCP) via AI Edge Portal Cloud API.
"""

from __future__ import annotations

import pathlib
import textwrap

import click


@click.command(
    "benchmark",
    help=textwrap.dedent("""\
        Benchmark LiteRT models on different platforms.

        MODEL: Path to the LiteRT model file.

        Examples:

          # Benchmark on Android with CPU (Default)

            $ litert benchmark model.tflite

          # Benchmark on Android with GPU

            $ litert benchmark model.tflite --gpu

          # Benchmark on Google AI Edge Portal in Google Cloud, using Pixel 7 devices.

            $ litert benchmark model.tflite --gcp --device "pixel 7"
        """),
)
@click.argument(
    "model",
    type=click.Path(
        exists=False, dir_okay=False, resolve_path=True, path_type=pathlib.Path
    ),
)
@click.option(
    "--android",
    "target",
    flag_value="android",
    default=True,
    help="Benchmark on Android device (Default)",
)
@click.option(
    "--gcp",
    "target",
    flag_value="gcp",
    help="Benchmark on Google AI Edge Portal in Google Cloud.",
)
@click.option(
    "--cpu",
    "accelerator",
    flag_value="cpu",
    default=True,
    help="Use CPU accelerator (Default)",
)
@click.option(
    "--gpu",
    "accelerator",
    flag_value="gpu",
    help="Use GPU accelerator",
)
@click.option(
    "--npu",
    "accelerator",
    flag_value="npu",
    help="Use NPU accelerator",
)
@click.option(
    "--device",
    type=str,
    default="pixel 7",
    help="Target device model (e.g., 'pixel 7'). Default is 'pixel 7'",
)
def benchmark_cmd(
    model: pathlib.Path,
    target: str,
    accelerator: str,
    device: str,
) -> None:
  """Benchmarks LiteRT models on different platforms.

  Args:
    model: Path to the LiteRT model file.
    target: Target platform for benchmark (android, gcp).
    accelerator: Accelerator to use (cpu, gpu, npu).
    device: Target device model (e.g., 'pixel 7').
  """
  if target == "android":
    # pylint: disable=g-import-not-at-top
    from litert_cli.commands.benchmark import android

    if not model.exists():
      click.secho(f"Error: Local model file not found: {model}", fg="red")
      return

    android.run_android(model_path=model, accelerator=accelerator)
  elif target == "gcp":
    # pylint: disable=g-import-not-at-top
    from litert_cli.commands.benchmark import gcp

    gcp.run_gcp(
        str(model),
        accelerator,
        device,
    )
  else:
    click.secho(f"Target '{target}' is not yet supported.", fg="red")
