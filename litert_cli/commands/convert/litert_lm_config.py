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

"""Canonical LiteRT-LM model configs for `litert convert <hf_id>`.

LiteRT-LM keeps a canonical `LlmMetadataProto.pbtext` (model type, Jinja chat
template, thought channel, stop tokens and sampler) per model family under
https://github.com/google-ai-edge/LiteRT-LM/tree/main/models. Without it, the
exported `.litertlm` falls back to a generic prefix/suffix prompt template, so
the runtime cannot toggle thinking or parse tool calls for the family.

This module maps a Hugging Face config to its family, fetches the canonical
metadata (latest `main` by default, cached locally), and rewrites
`max_num_tokens` to the KV cache length used for the conversion.
"""

from __future__ import annotations

import os
import pathlib
import re
import urllib.error
import urllib.request

# LiteRT-LM git ref the canonical configs are read from. Defaults to the latest
# configs on `main`, which match the litert-lm nightly the CLI installs. Set
# the LITERT_LM_MODELS_REF environment variable to a commit for reproducible
# conversions.
DEFAULT_LITERT_LM_MODELS_REF = "main"
_ENV_MODELS_REF = "LITERT_LM_MODELS_REF"
# A full or abbreviated commit SHA never changes, so its cache never goes stale.
_COMMIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")
_RAW_URL = (
    "https://raw.githubusercontent.com/google-ai-edge/LiteRT-LM/"
    "{ref}/models/{family}/LlmMetadataProto.pbtext"
)
_HTTP_TIMEOUT_SECS = 30

# Special values of `litert convert --llm-metadata`.
AUTO = "auto"
NONE = "none"

# Hugging Face `model_type` (top-level or text_config) -> LiteRT-LM family.
_MODEL_TYPE_TO_FAMILY = {
    "qwen3": "qwen3",
    "qwen2": "qwen2_5",
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "lfm2": "lfm2",
}
# Families whose checkpoints reuse a generic architecture (e.g. MiniCPM5 ships
# as `llama`), matched by the model id instead.
_MODEL_ID_PATTERNS = ((re.compile(r"minicpm5", re.IGNORECASE), "minicpm5"),)


def detect_family(model_id: str, config) -> str | None:
  """Returns the LiteRT-LM model family for a Hugging Face model, if known."""
  for pattern, family in _MODEL_ID_PATTERNS:
    if pattern.search(model_id) or pattern.search(
        str(getattr(config, "_name_or_path", "") or "")
    ):
      return family
  model_types = [getattr(config, "model_type", None)]
  text_config = getattr(config, "text_config", None)
  if text_config is not None:
    model_types.append(getattr(text_config, "model_type", None))
  for model_type in model_types:
    if model_type in _MODEL_TYPE_TO_FAMILY:
      return _MODEL_TYPE_TO_FAMILY[model_type]
  return None


def _cache_dir() -> pathlib.Path:
  base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
  return pathlib.Path(base) / "litert-cli" / "litert_lm_models"


def fetch_canonical_metadata(family: str, ref: str | None = None) -> str:
  """Returns the canonical LlmMetadataProto.pbtext text for `family`.

  A commit ref is served from the local cache once fetched. Moving refs such
  as `main` are fetched on every call; the cached copy is used only when the
  fetch fails (e.g. offline).

  Args:
    family: LiteRT-LM model family directory name (e.g. "qwen3").
    ref: LiteRT-LM git ref; defaults to LITERT_LM_MODELS_REF or `main`.

  Raises:
    RuntimeError: If the config cannot be fetched and is not cached.
  """
  ref = ref or os.environ.get(_ENV_MODELS_REF) or DEFAULT_LITERT_LM_MODELS_REF
  cached = _cache_dir() / ref / f"{family}.pbtext"
  if cached.exists() and _COMMIT_SHA.match(ref):
    return cached.read_text()
  url = _RAW_URL.format(ref=ref, family=family)
  try:
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_SECS) as resp:
      text = resp.read().decode("utf-8")
  except (urllib.error.URLError, OSError) as e:
    if cached.exists():
      return cached.read_text()
    raise RuntimeError(
        f"Could not fetch the canonical LiteRT-LM config for '{family}' from"
        f" {url}: {e}"
    ) from e
  cached.parent.mkdir(parents=True, exist_ok=True)
  cached.write_text(text)
  return text


def with_max_num_tokens(pbtext: str, max_num_tokens: int) -> str:
  """Sets the top-level `max_num_tokens` field of an LlmMetadata pbtext."""
  pattern = re.compile(r"^max_num_tokens:\s*\d+\s*$", re.MULTILINE)
  line = f"max_num_tokens: {max_num_tokens}"
  if pattern.search(pbtext):
    return pattern.sub(line, pbtext, count=1)
  return pbtext.rstrip("\n") + "\n" + line + "\n"


def write_metadata_override(
    family: str, cache_length: int, output_dir: str
) -> str:
  """Writes the canonical metadata for `family` and returns its path."""
  text = with_max_num_tokens(fetch_canonical_metadata(family), cache_length)
  path = pathlib.Path(output_dir) / f"litert_lm_{family}_metadata.pbtext"
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(text)
  return str(path)
