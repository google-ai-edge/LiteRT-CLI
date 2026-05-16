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

"""CLI Benchmark Module providing commands to benchmark LiteRT models.

This module defines the `benchmark` command using Click. It supports:
1. Benchmarking on a local Android device via adb (Default).
2. Benchmarking on Google Cloud Platform (GCP) via AI Edge Portal Cloud API.
"""

from __future__ import annotations

import pathlib
import textwrap

import click
from litert_cli.core import constants
from litert_cli.core import utils


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
    "--desktop",
    "target",
    flag_value="desktop",
    help="Benchmark on local Desktop machine.",
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
    "--jit",
    "compilation_mode",
    flag_value="jit",
    default=True,
    help="Use JIT (Just-in-time) compilation mode for NPU (Default).",
)
@click.option(
    "--aot",
    "compilation_mode",
    flag_value="aot",
    help="Use AOT (Ahead-of-time) compilation mode for NPU.",
)
@click.option(
    "--soc-model",
    type=str,
    default="SM8750",
    help="Target SoC model name for NPU AOT mode (e.g., 'SM8750').",
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
@click.option(
    "--gcp-bucket",
    type=str,
    help="GCS bucket name for uploading model (Only for GCP target).",
)
@click.option(
    "--num-runs",
    type=int,
    default=50,
    help="Target number of benchmark iterations. Default is 50.",
)
@click.option(
    "--warmup-runs",
    type=int,
    default=1,
    help="Number of warmup iterations before benchmarking. Default is 1.",
)
@click.option(
    "--min-secs",
    type=float,
    default=1.0,
    help="Minimum seconds to run. Default is 1.0.",
)
@click.option(
    "--max-secs",
    type=float,
    default=150.0,
    help="Maximum seconds to run. Default is 150.0.",
)
@click.option(
    "--warmup-min-secs",
    type=float,
    default=0.5,
    help="Minimum warmup duration in seconds. Default is 0.5.",
)
@click.option(
    "--input-layer-value-range",
    type=str,
    help=(
        "A map-like string representing value range for input layers (e.g."
        " input1,1.0,2.0:input2,0,254)."
    ),
)
@click.option(
    "--signature-key",
    type=str,
    help=(
        "The signature key to benchmark. If not specified, the default"
        " signature is used."
    ),
)
def benchmark_cmd(
    model: str,
    target: str,
    accelerator: str,
    devices: tuple[str, ...],
    compilation_mode: str,
    soc_model: str,
    gcp_project: str | None = None,
    gcp_bucket: str | None = None,
    num_runs: int = 50,
    warmup_runs: int = 1,
    min_secs: float = 1.0,
    max_secs: float = 150.0,
    warmup_min_secs: float = 0.5,
    input_layer_value_range: str | None = None,
    signature_key: str | None = None,
) -> None:
  """Benchmarks LiteRT models on different platforms.

  Args:
    model: Path to the LiteRT model file or Model Reference.
    target: Target platform for benchmark (android, gcp, desktop).
    accelerator: Accelerator to use (cpu, gpu, npu).
    devices: Target device model(s) (e.g., 'pixel 7').
    compilation_mode: Compilation mode for NPU (jit, aot).
    soc_model: Target SoC model for NPU AOT mode.
    gcp_project: GCP project ID for benchmarking.
    gcp_bucket: GCS bucket name for uploading model.
    num_runs: Target number of benchmark iterations.
    warmup_runs: Number of warmup iterations before benchmarking.
    min_secs: Minimum seconds to run.
    max_secs: Maximum seconds to run.
    warmup_min_secs: Minimum warmup duration in seconds.
    input_layer_value_range: Value range for input layers.
    signature_key: The signature key to benchmark.
  """
  from litert_cli.core import models as core_models

  # Quiet if default is true
  if constants.DEFAULT_QUIET:
    utils.enable_quiet_mode()

  resolved_model_path, _ = core_models.resolve_model_reference(model)

  if resolved_model_path != model:
    click.echo(f"Resolved model '{model}' to '{resolved_model_path}'")

  model_path = pathlib.Path(resolved_model_path)

  if target == "android":
    # pylint: disable=g-import-not-at-top
    from litert_cli.commands.benchmark import android

    if not model_path.exists():
      raise click.ClickException(f"Local model file not found: {model_path}")

    android.run_android(
        model_path=model_path,
        accelerator=accelerator,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        min_secs=min_secs,
        max_secs=max_secs,
        warmup_min_secs=warmup_min_secs,
        input_layer_value_range=input_layer_value_range,
        signature_key=signature_key,
    )
  elif target == "desktop":
    # pylint: disable=g-import-not-at-top
    from litert_cli.commands.benchmark import desktop

    if not model_path.exists():
      raise click.ClickException(f"Local model file not found: {model_path}")

    desktop.run_desktop(
        model_path=model_path,
        accelerator=accelerator,
        num_runs=num_runs,
        warmup_runs=warmup_runs,
        min_secs=min_secs,
        max_secs=max_secs,
        warmup_min_secs=warmup_min_secs,
        input_layer_value_range=input_layer_value_range,
        signature_key=signature_key,
    )
  elif target == "gcp":
    # pylint: disable=g-import-not-at-top
    from litert_cli.commands.benchmark import gcp

    if accelerator != "npu":
      compilation_mode = None
      soc_model = None

    gcp.run_gcp(
        str(model_path),
        accelerator,
        devices,
        gcp_project,
        gcp_bucket,
        compilation_mode,
        soc_model,
    )
  else:
    click.secho(f"Target '{target}' is not yet supported.", fg="red")
