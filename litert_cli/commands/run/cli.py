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

from collections.abc import Sequence
import textwrap

import click
from litert_cli.core import constants
from litert_cli.core import deps
from litert_cli.core import utils


@click.command(
    "run",
    help=textwrap.dedent("""\
        Run LiteRT models locally or on device.

        MODEL: Path to the LiteRT model (.tflite) or a Model Reference (e.g., nvidia/parakeet-ctc-0.6b).

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

            $ litert run model.tflite --print-tensors --sample-size 10

          8. Run with multiple accelerators (npu -> gpu -> cpu fallback):

            $ litert run model.tflite --npu --gpu --cpu

            OR explicitly:

            $ litert run model.tflite --accelerator npu,gpu,cpu
        """),
)
@deps.require_extra("run")
@click.argument("model", type=str)
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
    "--model-params",
    "model_params",
    multiple=True,
    help="Model specific parameters in format key=value.",
)
@click.option(
    "--model-help",
    is_flag=True,
    default=False,
    help="Show help specific to the matched model plugin.",
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
    "--accelerator",
    type=str,
    help="Comma-separated list of hardware accelerators (e.g. npu,gpu,cpu).",
)
@click.option(
    "--cpu",
    is_flag=True,
    help="Use CPU accelerator.",
)
@click.option(
    "--gpu",
    is_flag=True,
    help="Use GPU accelerator.",
)
@click.option(
    "--npu",
    is_flag=True,
    help="Use NPU accelerator.",
)
@click.option(
    "--signature-index",
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
    "--print-tensors",
    is_flag=True,
    default=False,
    help="Print output tensor values after execution.",
)
@click.option(
    "--sample-size",
    type=int,
    default=5,
    help="Number of sample elements to print from tensors. Default is 5.",
)
@click.pass_context
def run_cmd(
    unused_ctx: click.Context,
    model: str,
    inputs: Sequence[str],
    model_params: Sequence[str],
    model_help: bool,
    target: str,
    accelerator: str | None,
    cpu: bool,
    gpu: bool,
    npu: bool,
    signature_index: int,
    iterations: int,
    print_tensors: bool,
    sample_size: int,
) -> None:
  r"""Runs LiteRT models locally or on device.

  Args:
    unused_ctx: Click context.
    model: Path to the LiteRT model (.tflite).
    inputs: Tuple of input assignments (e.g., 'name=value' or just 'value').
    model_params: Model specific parameters.
    model_help: Show help specific to the matched model plugin.
    target: Execution target ('desktop' or 'android').
    accelerator: Hardware accelerator ('cpu', 'gpu', or 'npu').
    cpu: Use CPU accelerator.
    gpu: Use GPU accelerator.
    npu: Use NPU accelerator.
    signature_index: Index of model signature to run.
    iterations: Number of times to execute the model for benchmarking.
    print_tensors: Whether to print output tensor elements.
    sample_size: Number of sample elements to print from tensors.
  """
  # Resolve the order of accelerators
  accelerator_list = []
  if accelerator:
    accelerator_list = [a.strip().lower() for a in accelerator.split(",") if a.strip()]
  else:
    if npu:
      accelerator_list.append("npu")
    if gpu:
      accelerator_list.append("gpu")
    if cpu:
      accelerator_list.append("cpu")

    if not accelerator_list:
      accelerator_list = ["cpu"]

  accelerator = ",".join(accelerator_list)

  # Quiet if default is true
  if constants.DEFAULT_QUIET:

    utils.enable_quiet_mode()

  # --- Model Reference and Cache Resolution ---
  from litert_cli.core import models as core_models  # pylint: disable=g-import-not-at-top

  resolved_model_path, resolved_hf_id = core_models.resolve_model_reference(
      model
  )

  if resolved_model_path != model:
    click.echo(f"Resolved model '{model}' to '{resolved_model_path}'")

  # --- Plugin Dispatch Mechanism ---
  # Try to delegate to a model-specific plugin first.
  from litert_cli import models  # pylint: disable=g-import-not-at-top

  # Parse model-params into a dictionary
  parsed_model_params = {}
  if model_params:
    for p in model_params:
      if "=" in p:
        k, v = p.split("=", 1)
        parsed_model_params[k] = v

  # Pass the resolved hf_id as model_id to dispatch, and the actual file path
  # in kwargs
  plugin_result = models.dispatch_model_intent(
      intent="run",
      model_id=resolved_hf_id or str(model),
      inputs=inputs,
      model_help=model_help,
      model_params=parsed_model_params,
      target=target,
      accelerator=accelerator,
      model_path=resolved_model_path,  # Pass the actual file path here!
  )

  if plugin_result is not None:
    # If the plugin handled it or showed help, we exit
    return
  # ----------------------------------

  if target == "desktop":
    from litert_cli.commands.run import desktop  # pylint: disable=g-import-not-at-top

    desktop.run_desktop(
        model_path=str(resolved_model_path),
        inputs=inputs,
        accelerator=accelerator,
        signature_index=signature_index,
        iterations=iterations,
        print_tensors=print_tensors,
        sample_size=sample_size,
    )
  elif target == "android":
    from litert_cli.commands.run import android  # pylint: disable=g-import-not-at-top

    android.run_android(
        model_path=str(resolved_model_path),
        inputs=inputs,
        accelerator=accelerator,
        signature_index=signature_index,
        iterations=iterations,
        print_tensors=print_tensors,
        sample_size=sample_size,
    )
  else:
    click.secho(f"Target '{target}' is not yet supported.", fg="red")
