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
    help="""\b
Benchmark LiteRT models on different platforms.
\b
MODEL: Path to the LiteRT model file.
\b
Examples:
\b
  # Benchmark on Desktop with CPU (Default) or GPU
\b
    $ litert benchmark model.tflite
    $ litert benchmark model.tflite --gpu
\b
  # Benchmark on Android with CPU (Default) or GPU or NPU
    $ litert benchmark model.tflite --android
    $ litert benchmark model.tflite --android --gpu
    $ litert benchmark model.tflite --android --npu
\b
  # Benchmark on Google AI Edge Portal in Google Cloud. Prerequisites:
  # - Set up your Google AI Edge Portal account by following up the instructions at:
  #   https://ai.google.dev/edge/ai-edge-portal
  # - Set up authentication by running: gcloud auth login
  # - You can set the default GCP project by setting the environment variable LITERT_GCP_PROJECT,
  #  or by providing the --gcp-project option.
  # 
    $ litert benchmark model.tflite --gcp --device "pixel 7"
    $ litert benchmark model.tflite --gcp --device "pixel 7" --gcp-project "your-gcp-project-id"
    $ litert benchmark model.tflite --gcp --devices "pixel 7, sm-s931u1" --gpu
""",
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
    "--devices",
    "devices",
    type=str,
    multiple=True,
    default=["pixel 7"],
    help=(
        "Target device model name(s) (e.g., 'pixel 7'). Can be specified"
        " --device multiple times or use --devices 'pixel 7, sm-s931u1'."
        " Default is 'pixel 7'"
    ),
)
@click.option(
    "--gcp-project",
    type=str,
    help="GCP project ID for benchmarking (Only for GCP target).",
)
def benchmark_cmd(
    model: str,
    target: str,
    accelerator: str,
    devices: tuple[str, ...],
    gcp_project: str | None = None,
) -> None:
  """Benchmarks LiteRT models on different platforms.

  Args:
    model: Path to the LiteRT model file or Model Reference.
    target: Target platform for benchmark (android, gcp).
    accelerator: Accelerator to use (cpu, gpu, npu).
    devices: Target device model(s) (e.g., 'pixel 7').
    gcp_project: GCP project ID for benchmarking.
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
        devices,
        gcp_project,
    )
  else:
    click.secho(f"Target '{target}' is not yet supported.", fg="red")
