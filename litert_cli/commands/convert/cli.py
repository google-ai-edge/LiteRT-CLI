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

"""CLI module for the `litert convert` command.

This command handles converting PyTorch and Hugging Face models into LiteRT
models using various conversion paths like automated HF export, generic script
injection, and native generative API re-authoring.
"""

from __future__ import annotations

import pathlib
import textwrap

import click
from litert_cli.core import deps


@click.command(
    "convert",
    help=textwrap.dedent("""\
        Convert a PyTorch model into a LiteRT model.

        MODEL_OR_SCRIPT: Hugging Face model ID or path to a PyTorch script.

        Examples:

          Automated HF Conversion:

            $ litert convert Qwen/Qwen1.5-0.5B-Chat --output /tmp/qwen

          Generic Script Injection:

            $ litert convert my_model.py --output /tmp/mymodel
    """),
)
@deps.require_extra("convert")
@click.argument("model_or_script", type=str, required=True)
@click.option(
    "--output",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=False,
    default=None,
    help="Directory to save the converted TFLite model.",
)
@click.option(
    "--model-func",
    type=str,
    default="get_model",
    help=(
        "Name of the function in the --script that returns a torch.nn.Module"
        " and optionally a quantization config. Default: 'get_model'."
    ),
)
@click.option(
    "--input-func",
    type=str,
    default="get_args",
    help=(
        "Name of the function in the --script that returns (True) sample_args "
        "and/or kwargs. Default: 'get_args'."
    ),
)
@click.option(
    "--target",
    type=str,
    multiple=True,
    help=(
        "One or more NPU target codenames (e.g., sm8450) to apply AOT"
        " compilation."
    ),
)
@click.option(
    "--export-aipack",
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    required=False,
    default=None,
    help=(
        "If specified, exports an AI Pack directory for PODAI alongside the"
        " compiled model."
    ),
)
@click.option(
    "--quantize",
    type=str,
    default=None,
    help="Quantization recipe to apply (e.g., dynamic_wi8_afp32, fp16, int8_dynamic).",
)
@click.option(
    "--model-args",
    type=str,
    default=None,
    help="Comma-separated key=value arguments to pass to custom model/input functions.",
)
@click.option(
    "--prefill-lengths",
    type=str,
    default="256",
    help="Comma-separated list of prefill lengths for HuggingFace models. Default: '256'.",
)
@click.option(
    "--cache-length",
    type=int,
    default=4096,
    help="KV cache length for HuggingFace models. Default: 4096.",
)
@click.option(
    "--bundle-litert-lm/--no-bundle-litert-lm",
    is_flag=True,
    default=True,
    help="Bundle exported artifacts into a .litert_lm package (HuggingFace mode only). Default: True.",
)
def convert_cmd(
    model_or_script: str,
    output: pathlib.Path | None,
    model_func: str,
    input_func: str,
    target: tuple[str, ...],
    export_aipack: pathlib.Path | None,
    quantize: str | None,
    model_args: str | None,
    prefill_lengths: str,
    cache_length: int,
    bundle_litert_lm: bool,
) -> None:
  r"""Converts a PyTorch model into a LiteRT model.

  Args:
    model_or_script: Hugging Face model ID or path to a PyTorch script.
    output: Output directory for the converted model.
    model_func: Function to retrieve the model in 'script' mode.
    input_func: Function to retrieve sample inputs in 'script' mode.
    target: NPU targets to compile for.
    export_aipack: Output directory to export the AI Pack for PODAI.
    quantize: Quantization recipe to apply.
    model_args: Arguments to pass to custom model/input functions.
    prefill_lengths: List of prefill lengths for HuggingFace models.
    cache_length: KV cache length for HuggingFace models.
    bundle_litert_lm: Whether to bundle artifacts into a .litert_lm package.
  """

  from litert_cli.core import constants, utils
  import warnings

  if constants.DEFAULT_QUIET:
    utils.enable_quiet_mode()

  # Suppress noisy warnings from torch, torchao, etc.
  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings("ignore", category=SyntaxWarning)
  warnings.filterwarnings("ignore", category=UserWarning)

  if output is None:
    if model_or_script.endswith(".py"):
      base_name = pathlib.Path(model_or_script).stem
    else:
      base_name = pathlib.Path(model_or_script).name
    output = pathlib.Path.cwd() / base_name

  if constants.ENABLE_MODEL_PLUGINS:
    from litert_cli.models import dispatch_model_intent

    plugin_result = dispatch_model_intent(
        "convert",
        model_or_script,
        output=output,
        target=target,
        quantize=quantize,
        export_aipack=export_aipack,
        model_args=model_args,
        prefill_lengths=prefill_lengths,
        cache_length=cache_length,
        bundle_litert_lm=bundle_litert_lm,
    )
    if plugin_result is not None:
      return

  if model_or_script.endswith(".py"):
    from litert_cli.commands.convert import generic  # pylint: disable=g-import-not-at-top

    generic.convert_generic_script(
        model_or_script,
        model_func,
        input_func,
        str(output),
        target,
        export_aipack,
        quantize,
        model_args,
    )
  else:
    from litert_cli.commands.convert import huggingface  # pylint: disable=g-import-not-at-top

    huggingface.convert_huggingface(
        model_or_script,
        str(output),
        target,
        export_aipack,
        quantize,
        prefill_lengths,
        cache_length,
        bundle_litert_lm,
    )
