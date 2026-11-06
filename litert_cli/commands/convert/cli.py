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

# Options that apply to one conversion mode only.
_SCRIPT_ONLY_OPTIONS = (
    "model_func",
    "input_func",
    "export_aipack",
    "model_args",
)
_HF_ONLY_OPTIONS = (
    "prefill_lengths",
    "cache_length",
    "bundle_litert_lm",
    "llm_metadata",
    "task",
    "dry_run",
)


def _validate_prefill_lengths(value: str | None) -> str | None:
  """Checks that --prefill-lengths is a list of positive integers."""
  if value is None:
    return None
  for item in value.split(","):
    if not item.strip().isdigit() or int(item) <= 0:
      raise click.BadParameter(
          f"'{value}' is not a comma-separated list of positive integers.",
          param_hint="--prefill-lengths",
      )
  return value


def _check_options_for_mode(
    ctx: click.Context,
    *,
    is_script: bool,
    target: tuple[str, ...],
    export_aipack: pathlib.Path | None,
) -> None:
  """Rejects options that the detected conversion mode would ignore.

  Args:
    ctx: The click context of the convert command.
    is_script: Whether MODEL_OR_SCRIPT is a PyTorch script.
    target: The --target values.
    export_aipack: The --export-aipack value.

  Raises:
    click.UsageError: If an option of the other mode was given.
  """
  other_mode = _HF_ONLY_OPTIONS if is_script else _SCRIPT_ONLY_OPTIONS
  given = [
      "--" + name.replace("_", "-")
      for name in other_mode
      if ctx.get_parameter_source(name)
      not in (None, click.core.ParameterSource.DEFAULT)
  ]
  if given:
    mode = "a PyTorch script" if is_script else "a Hugging Face model"
    raise click.UsageError(
        f"{', '.join(given)} cannot be used when converting {mode}."
    )
  if is_script and export_aipack and not target:
    raise click.UsageError(
        "--export-aipack needs --target: an AI Pack holds models compiled for"
        " NPU targets."
    )


@click.command(
    "convert",
    help=textwrap.dedent("""\
        Convert a PyTorch model into a LiteRT model.

        MODEL_OR_SCRIPT: Hugging Face model ID, local checkpoint directory, or
        path to a PyTorch script.

        Examples:

          Automated HF Conversion:

            $ litert convert Qwen/Qwen3-0.6B --output /tmp/qwen

          Print the resolved export plan without exporting:

            $ litert convert Qwen/Qwen3-0.6B --dry-run

          HF Conversion with Weight-Only INT4 Quantization:

            $ litert convert Qwen/Qwen3-0.6B --quantize weight_only_wi4_afp32 --output /tmp/qwen_w4

          HF Conversion with Custom Prefill & Cache Lengths:

            $ litert convert Qwen/Qwen3-0.6B --prefill-lengths "128,512" --cache-length 2048 --output /tmp/qwen_custom

          HF Conversion for NPU (uncompiled; compile it for a device later):

            $ litert convert Qwen/Qwen3-0.6B --target npu:qualcomm --output /tmp/qwen_npu

          Generic Script Injection with Quantization:

            $ litert convert my_model.py --quantize-recipe dynamic_wi8_afp32 --output /tmp/mymodel

          Generic Script with Model Args and AOT Target Compilation:

            $ litert convert my_model.py --model-args "batch_size=4" --target sm8750 --output /tmp/mymodel_npu

        Options marked "(script mode)" apply to a PyTorch script only, and
        "(HF mode)" to a Hugging Face model or checkpoint only; passing one in
        the other mode is an error.
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
        "(script mode) Name of the function in the script that returns a"
        " torch.nn.Module and optionally a quantization config."
    ),
)
@click.option(
    "--input-func",
    type=str,
    default="get_args",
    help=(
        "(script mode) Name of the function in the script that returns the"
        " sample args (a tuple) or kwargs (a dict)."
    ),
)
@click.option(
    "--target",
    type=str,
    multiple=True,
    help=(
        "HF mode: one export target, 'cpu_gpu' (default), 'cpu', 'gpu', 'npu'"
        " or 'npu:<vendor>' (e.g. npu:qualcomm; the model is not AOT-compiled)."
        " Script mode: one or more NPU SoC codenames (e.g. sm8750) to"
        " AOT-compile for; repeat the option for several."
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
        "(script mode, with --target) Export the compiled models as an AI Pack"
        " directory for Play for On-device AI instead of .tflite files."
    ),
)
@click.option(
    "--quantize",
    "--quantize-recipe",
    type=str,
    default=None,
    help=(
        "Quantization recipe (e.g. dynamic_wi8_afp32, weight_only_wi4_afp32;"
        " 'none' for no quantization). Alias: --quantize-recipe. HF mode"
        " accepts the litert_quantizer recipes; script mode accepts"
        " dynamic_wi8_afp32, dynamic_wi4_afp32, weight_only_wi8_afp32,"
        " weight_only_wi4_afp32, fp16 and pt2e_{static,dynamic}[_per_channel]."
    ),
)
@click.option(
    "--model-args",
    type=str,
    default=None,
    help=(
        "(script mode) Comma-separated key=value arguments for the model and"
        " input functions; values are parsed as Python literals."
    ),
)
@click.option(
    "--prefill-lengths",
    type=str,
    default=None,
    callback=lambda ctx, param, value: _validate_prefill_lengths(value),
    help=(
        "(HF mode) Comma-separated prefill lengths. Default: litert-torch's"
        " recommendation for the model/target."
    ),
)
@click.option(
    "--cache-length",
    type=int,
    default=None,
    help=(
        "(HF mode) KV cache length. Default: litert-torch's recommendation for"
        " the model/target."
    ),
)
@click.option(
    "--bundle-litert-lm/--no-bundle-litert-lm",
    default=None,
    help=(
        "(HF mode) Bundle the exported artifacts into a .litertlm file."
        " Default: litert-torch's recommendation for the model."
    ),
)
@click.option(
    "--llm-metadata",
    type=str,
    default="auto",
    show_default=True,
    help=(
        "(HF mode) LlmMetadata (chat template, stop tokens, thinking/tool-call"
        " config) for the .litertlm bundle. 'auto' uses the"
        " canonical config from LiteRT-LM's models/<family> for supported"
        " families (Qwen2.5, Qwen3, Qwen3.5, LFM2, MiniCPM5); 'none' keeps the"
        " metadata generated by export_hf; otherwise a path to an"
        " LlmMetadataProto pbtext."
    ),
)
@click.option(
    "--task",
    type=str,
    default="auto",
    show_default=True,
    help=(
        "(HF mode) Export task ('auto' to detect from the checkpoint,"
        " 'text_generation', 'image_text_to_text', 'multimodal_lm',"
        " 'automatic_speech_recognition', 'text_to_speech', 'text_to_image',"
        " or 'any_to_any')."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="(HF mode) Print the resolved export plan and exit.",
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
    prefill_lengths: str | None,
    cache_length: int | None,
    bundle_litert_lm: bool | None,
    llm_metadata: str,
    task: str,
    dry_run: bool,
) -> None:
  r"""Converts a PyTorch model into a LiteRT model.

  Args:
    model_or_script: Hugging Face model ID or path to a PyTorch script.
    output: Output directory for the converted model.
    model_func: Function to retrieve the model in 'script' mode.
    input_func: Function to retrieve sample inputs in 'script' mode.
    target: Hardware target for HF export or NPU targets to compile for.
    export_aipack: Output directory to export the AI Pack for PODAI.
    quantize: Quantization recipe to apply (see litert_quantizer.recipe for full
      list).
    model_args: Arguments to pass to custom model/input functions.
    prefill_lengths: List of prefill lengths for HuggingFace models.
    cache_length: KV cache length for HuggingFace models.
    bundle_litert_lm: Whether to bundle artifacts into a .litert_lm package.
    llm_metadata: 'auto', 'none' or a path to an LlmMetadataProto pbtext.
    task: HuggingFace export task, or 'auto' to detect it.
    dry_run: Only print the resolved export plan.
  """

  from litert_cli.core import constants, utils
  import warnings

  if constants.DEFAULT_QUIET:
    utils.enable_quiet_mode()

  # Suppress noisy warnings from torch, torchao, etc.
  warnings.filterwarnings("ignore", category=FutureWarning)
  warnings.filterwarnings("ignore", category=SyntaxWarning)
  warnings.filterwarnings("ignore", category=UserWarning)

  _check_options_for_mode(
      click.get_current_context(),
      is_script=model_or_script.endswith(".py"),
      target=target,
      export_aipack=export_aipack,
  )

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
        bundle_litert_lm=(
            True if bundle_litert_lm is None else bundle_litert_lm
        ),
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
        llm_metadata,
        task=task,
        dry_run=dry_run,
    )
