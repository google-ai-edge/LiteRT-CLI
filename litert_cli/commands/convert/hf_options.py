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

"""High-level option parsing for `litert convert <hf_id>`.

These helpers only translate the CLI's generic options into litert-torch
`export_hf` keyword arguments. They hold no model-specific knowledge: the task,
recommended export settings and LiteRT-LM chat template of each architecture
come from litert-torch (`auto_export` and its `ModelExtension`s).

This module only depends on the standard library so it can be tested without
importing torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

# Value of `--task` that lets litert-torch detect the task.
TASK_AUTO = "auto"
# litert-torch `export_hf` tasks, plus short aliases accepted by `--task`.
TASKS = (
    "text_generation",
    "image_text_to_text",
    "automatic_speech_recognition",
    "text_to_speech",
    "text_to_image",
    "any_to_any",
)
TASK_ALIASES = {
    "llm": "text_generation",
    "text": "text_generation",
    "vlm": "image_text_to_text",
    "asr": "automatic_speech_recognition",
    "speech2text": "automatic_speech_recognition",
    "stt": "automatic_speech_recognition",
    "tts": "text_to_speech",
    "text2speech": "text_to_speech",
    "t2i": "text_to_image",
    "text2image": "text_to_image",
    "omni": "any_to_any",
}
# Tasks whose output is a LiteRT-LM LLM bundle.
LLM_TASKS = ("text_generation", "image_text_to_text")
# Values of `--quantize` (and `--export-arg *quantization_recipe=`) that mean
# float weights. export_hf drops `None` arguments and falls back to its default
# recipe, so float is passed as ''.
FLOAT_QUANTIZE_VALUES = ("none", "float", "fp32", "")
# Arguments the CLI sets itself; `--export-arg` must not override them.
_RESERVED_EXPORT_ARGS = {
    "model": "the MODEL_OR_SCRIPT argument",
    "output_dir": "--output",
    "task": "--task",
}


def normalize_task(task: str | None) -> str | None:
  """Maps a `--task` value or alias to an export_hf task.

  Args:
    task: The `--task` value; `None` or 'auto' for automatic detection.

  Returns:
    The export_hf task name, or `None` to let litert-torch detect it.

  Raises:
    ValueError: If `task` is not a known task or alias.
  """
  if task is None or task.strip().lower() == TASK_AUTO:
    return None
  name = task.strip().lower()
  name = TASK_ALIASES.get(name, name)
  if name not in TASKS:
    raise ValueError(
        f"Unknown task '{task}'. Use one of: {TASK_AUTO}, {', '.join(TASKS)}"
        f" (or {', '.join(sorted(TASK_ALIASES))})."
    )
  return name


def parse_quantize(value: str | None) -> str | None:
  """Returns the export_hf quantization recipe ('' for float, None if unset)."""
  if value is None:
    return None
  if value.strip().lower() in FLOAT_QUANTIZE_VALUES:
    return ""
  return value.strip()


def parse_prefill_lengths(value: str | None) -> list[int] | None:
  """Parses a comma-separated `--prefill-lengths` value."""
  if value is None:
    return None
  try:
    return [int(length) for length in value.split(",") if length.strip()]
  except ValueError:
    raise ValueError(
        f"--prefill-lengths expects comma-separated integers, got '{value}'."
    ) from None


def _parse_scalar(raw: str) -> Any:
  """Converts a `--export-arg` value: bool, None, int, float, JSON, else str."""
  lowered = raw.lower()
  if lowered in ("true", "false"):
    return lowered == "true"
  if lowered in ("none", "null"):
    return None
  for cast in (int, float):
    try:
      return cast(raw)
    except ValueError:
      pass  # Not this numeric type; try the next one.
  if raw.startswith(("{", "[")):
    try:
      return json.loads(raw)
    except ValueError:
      pass  # Not valid JSON; keep the raw string.
  return raw


def parse_export_args(values: Sequence[str]) -> dict[str, Any]:
  """Parses repeated `--export-arg KEY=VALUE` options.

  Args:
    values: The raw `KEY=VALUE` strings.

  Returns:
    export_hf keyword arguments. Quantization recipes set to a float alias
    become '' (float); `none` for other keys becomes `None` (export_hf default).

  Raises:
    ValueError: If an item is not `KEY=VALUE` or sets an argument that has its
      own CLI option.
  """
  result = {}
  for item in values:
    if "=" not in item:
      raise ValueError(f"--export-arg expects KEY=VALUE, got '{item}'.")
    key, raw = item.split("=", 1)
    key = key.strip().lstrip("-").replace("-", "_")
    if key in _RESERVED_EXPORT_ARGS:
      raise ValueError(
          f"--export-arg cannot set '{key}'; use {_RESERVED_EXPORT_ARGS[key]}."
      )
    raw = raw.strip()
    if key.endswith("quantization_recipe"):
      result[key] = parse_quantize(raw)
    else:
      result[key] = _parse_scalar(raw)
  return result


def build_overrides(
    *,
    quantize: str | None = None,
    prefill_lengths: str | None = None,
    cache_length: int | None = None,
    bundle_litert_lm: bool | None = None,
    gpu_optimizations: bool | None = None,
    externalize_weights: bool | None = None,
    export_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Returns the export_hf arguments set explicitly on the command line.

  Unset options are left out, so litert-torch's recommended defaults apply.
  `--export-arg` values win over the dedicated options.

  Args:
    quantize: `--quantize` value.
    prefill_lengths: `--prefill-lengths` value.
    cache_length: `--cache-length` value.
    bundle_litert_lm: `--bundle-litert-lm` / `--no-bundle-litert-lm`.
    gpu_optimizations: `--gpu-optimizations` / `--no-gpu-optimizations`.
    externalize_weights: `--externalize-weights` / `--no-externalize-weights`.
    export_args: Parsed `--export-arg` values.

  Returns:
    export_hf keyword arguments.
  """
  overrides = {
      "quantization_recipe": parse_quantize(quantize),
      "prefill_lengths": parse_prefill_lengths(prefill_lengths),
      "cache_length": cache_length,
      "bundle_litert_lm": bundle_litert_lm,
      "apply_gpu_composites": gpu_optimizations,
      "externalize_weights": externalize_weights,
  }
  overrides = {k: v for k, v in overrides.items() if v is not None}
  overrides.update(export_args or {})
  return overrides
