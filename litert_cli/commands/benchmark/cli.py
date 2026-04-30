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
@click.argument("model", type=str)
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
    model: str,
    target: str,
    accelerator: str,
    device: str,
) -> None:
  """Benchmarks LiteRT models on different platforms.

  Args:
    model: Path to the LiteRT model file or Model Reference.
    target: Target platform for benchmark (android, gcp).
    accelerator: Accelerator to use (cpu, gpu, npu).
    device: Target device model (e.g., 'pixel 7').
  """
  from litert_cli.core import models as core_models

  resolved_model_path, _ = core_models.resolve_model_reference(model)

  if resolved_model_path != model:
    click.echo(f"Resolved model '{model}' to '{resolved_model_path}'")

  model_path = pathlib.Path(resolved_model_path)

  if target == "android":
    # pylint: disable=g-import-not-at-top
    from litert_cli.commands.benchmark import android

    if not model_path.exists():
      raise click.ClickException(f"Local model file not found: {model_path}")

    android.run_android(model_path=model_path, accelerator=accelerator)
  elif target == "gcp":
    # pylint: disable=g-import-not-at-top
    from litert_cli.commands.benchmark import gcp

    gcp.run_gcp(
        str(model_path),
        accelerator,
        device,
    )
  else:
    click.secho(f"Target '{target}' is not yet supported.", fg="red")
