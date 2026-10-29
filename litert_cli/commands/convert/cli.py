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

        MODEL_OR_SCRIPT: Hugging Face model ID, local checkpoint directory, or
        path to a PyTorch script.

        For Hugging Face models, litert-torch detects the architecture and
        applies the export settings recommended for it (task, quantization,
        prefill/cache lengths, GPU composites, externalized weights, LiteRT-LM
        chat template), so one command produces a ready-to-run model. Every
        value can be overridden with a flag; the "Export plan" printed at the
        start shows each value and where it came from. Any architecture with a
        litert-torch ModelExtension is supported, including out-of-tree ones
        loaded with --extension-modules.

        Examples:

        \b
          LLM with the recommended settings (GPU composites, 128+1024 prefill):
            $ litert convert Qwen/Qwen3-0.6B

        \b
          Show the plan without exporting:
            $ litert convert LiquidAI/LFM2.5-1.2B-Instruct --dry-run

        \b
          Speech-to-text / text-to-speech / text-to-image:
            $ litert convert openai/whisper-tiny
            $ litert convert hexgrad/Kokoro-82M
            $ litert convert black-forest-labs/FLUX.2-klein-4B

        \b
          Override recommended values:
            $ litert convert Qwen/Qwen3-1.7B --quantize dynamic_wi8_afp32 \\
                --prefill-lengths 128,512 --cache-length 4096 \\
                --no-gpu-optimizations

        \b
          Pass any other litert-torch export_hf flag:
            $ litert convert Qwen/Qwen3-0.6B \\
                --export-arg use_sdpa_composite=False

        \b
          Generic script with model args and AOT target compilation:
            $ litert convert my_model.py --model-args "batch_size=4" \\
                --target sm8450
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
    "--quantize-recipe",
    type=str,
    default=None,
    help=(
        "Quantization recipe to apply (e.g., dynamic_wi8_afp32,"
        " dynamic_wi4b32_afp32, weight_only_wi4_afp32; 'none' for float)."
        " Alias: --quantize-recipe. Default for HuggingFace models:"
        " litert-torch's recommendation for the model. For the full list, see"
        " 'litert_quantizer.recipe'."
    ),
)
@click.option(
    "--model-args",
    type=str,
    default=None,
    help=(
        "Comma-separated key=value arguments to pass to custom model/input"
        " functions."
    ),
)
@click.option(
    "--prefill-lengths",
    type=str,
    default=None,
    help=(
        "Comma-separated list of prefill lengths for HuggingFace LLMs."
        " Default: litert-torch's recommendation ('128,1024')."
    ),
)
@click.option(
    "--cache-length",
    type=int,
    default=None,
    help=(
        "KV cache length for HuggingFace LLMs. Default: litert-torch's"
        " recommendation (32768)."
    ),
)
@click.option(
    "--bundle-litert-lm/--no-bundle-litert-lm",
    default=None,
    help=(
        "Bundle exported artifacts into a .litertlm package (HuggingFace mode"
        " only). Default: litert-torch's choice for the model."
    ),
)
@click.option(
    "--llm-metadata",
    type=str,
    default="auto",
    show_default=True,
    help=(
        "LlmMetadata (chat template, stop tokens, thinking/tool-call config)"
        " for the .litertlm bundle in HuggingFace mode. 'auto' uses the"
        " canonical config from LiteRT-LM's models/<family> when litert-torch"
        " names one for the model; 'none' keeps the"
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
        "HuggingFace export task: auto, text_generation (llm),"
        " image_text_to_text (vlm), automatic_speech_recognition (asr),"
        " text_to_speech (tts), text_to_image (t2i) or any_to_any (omni)."
    ),
)
@click.option(
    "--gpu-optimizations/--no-gpu-optimizations",
    default=None,
    help=(
        "Emit GPU composites, fused QKV/gate-up projections, a boolean mask"
        " and a GPU-friendly cache length (litert-torch"
        " apply_gpu_composites). The model still runs on CPU. Default: on"
        " for LLMs and VLMs."
    ),
)
@click.option(
    "--externalize-weights/--no-externalize-weights",
    default=None,
    help=(
        "Move the TFLite weights into one shared tflite_weights section of"
        " the .litertlm. Default: on for LLMs and VLMs."
    ),
)
@click.option(
    "--extension-modules",
    type=str,
    default=None,
    help=(
        "Comma-separated Python modules (or .py files, with a litert-torch"
        " that supports them) registering out-of-tree ModelExtension classes."
    ),
)
@click.option(
    "--export-arg",
    "export_args",
    type=str,
    multiple=True,
    metavar="KEY=VALUE",
    help=(
        "Extra litert-torch export_hf argument, repeatable (e.g."
        " --export-arg use_sdpa_composite=False --export-arg input_sec=10)."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the resolved export plan for a HuggingFace model and exit.",
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
    gpu_optimizations: bool | None,
    externalize_weights: bool | None,
    extension_modules: str | None,
    export_args: tuple[str, ...],
    dry_run: bool,
) -> None:
  r"""Converts a PyTorch model into a LiteRT model.

  Args:
    model_or_script: Hugging Face model ID or path to a PyTorch script.
    output: Output directory for the converted model.
    model_func: Function to retrieve the model in 'script' mode.
    input_func: Function to retrieve sample inputs in 'script' mode.
    target: NPU targets to compile for.
    export_aipack: Output directory to export the AI Pack for PODAI.
    quantize: Quantization recipe to apply (see litert_quantizer.recipe for
      full list).
    model_args: Arguments to pass to custom model/input functions.
    prefill_lengths: List of prefill lengths for HuggingFace models.
    cache_length: KV cache length for HuggingFace models.
    bundle_litert_lm: Whether to bundle a .litertlm package (None: litert-torch
      default for the model).
    llm_metadata: 'auto', 'none' or a path to an LlmMetadataProto pbtext.
    task: HuggingFace export task, or 'auto' to detect it.
    gpu_optimizations: Whether to emit GPU composites (None: recommended).
    externalize_weights: Whether to externalize weights (None: recommended).
    extension_modules: Comma-separated module names or .py paths for extensions.
    export_args: Extra KEY=VALUE export_hf arguments.
    dry_run: Only print the export plan.
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
        gpu_optimizations=gpu_optimizations,
        externalize_weights=externalize_weights,
        extension_modules=extension_modules,
        export_args=export_args,
        dry_run=dry_run,
    )
