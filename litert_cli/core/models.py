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

"""Model plugin for LiteRT CLI."""

from __future__ import annotations

import json
import os
import pathlib

import click
from litert_cli.core import constants


def ref_dir_name(model_ref: str) -> str | None:
  """Returns the cache directory name of a model reference, or None.

  The main part of the reference (before any `:sub`) is flattened (`a/b` ->
  `a__b`). Returns None for a name that would escape the models cache, such as
  an empty name, `.`, `..` or one with a path separator left.

  Args:
    model_ref: The model reference, optionally with a `:sub` suffix.
  """
  main_ref = model_ref.split(":", 1)[0]
  name = main_ref.replace("/", "__")
  if not name or name in (".", "..") or os.sep in name or "\\" in name:
    return None
  if os.altsep and os.altsep in name:
    return None
  return name


def cache_dir_for_ref(model_ref: str) -> pathlib.Path:
  """Returns the cache directory of a model reference.

  Args:
    model_ref: The model reference, optionally with a `:sub` suffix.

  Raises:
    click.BadParameter: If the reference is not a valid model reference name.
  """
  name = ref_dir_name(model_ref)
  if name is None:
    raise click.BadParameter(
        f"'{model_ref}' is not a valid model reference name.",
        param_hint="model reference",
    )
  return pathlib.Path(constants.LITERT_MODELS_CACHE_DIR) / name


def resolve_model_reference(model_ref: str) -> tuple[str, str | None]:
  """Resolves a model reference (path, HF ID, or cache ID) to a local path.

  Args:
    model_ref: The model identifier or path to resolve.

  Returns:
    A tuple of (resolved_model_path, resolved_hf_id).
  """
  resolved_model_path = str(model_ref)
  resolved_hf_id = None

  # Handle sub-reference format: ref:sub_ref (e.g. nvidia/parakeet:enc)
  main_ref = model_ref
  sub_ref = None

  if ":" in model_ref:
    parts = model_ref.split(":", 1)
    main_ref = parts[0]
    sub_ref = parts[1]

  ref_name = ref_dir_name(main_ref)
  if ref_name is None:
    return resolved_model_path, resolved_hf_id
  cache_path = pathlib.Path(constants.LITERT_MODELS_CACHE_DIR) / ref_name

  # 1. Check if it's a reference in the cache
  if not (cache_path.exists() and cache_path.is_dir()):
    return resolved_model_path, resolved_hf_id

  metadata_file = cache_path / "metadata.json"

  if not metadata_file.exists():
    # Default behavior: find the first .tflite file in the directory
    tflite_files = list(cache_path.glob("*.tflite"))
    if tflite_files:
      resolved_model_path = str(tflite_files[0])
    else:
      resolved_model_path = str(cache_path)
    return resolved_model_path, resolved_hf_id

  try:
    with open(metadata_file, "r") as f:
      metadata = json.load(f)
      resolved_hf_id = metadata.get("hf_id")

      # Handle sub-reference resolution
      if sub_ref:
        sub_refs = metadata.get("sub_references", {})
        if sub_ref in sub_refs:
          sub_info = sub_refs[sub_ref]
          file_name = sub_info.get("file")
          if file_name:
            resolved_model_path = str(cache_path / file_name)
            if "hf_id" in sub_info:
              resolved_hf_id = sub_info["hf_id"]
            return resolved_model_path, resolved_hf_id

        # Fallback: Check if sub_ref is a literal filename in the directory
        potential_file = cache_path / sub_ref
        if potential_file.exists() and potential_file.is_file():
          return str(potential_file), resolved_hf_id

      # Default behavior: find the first .tflite file in the directory
      tflite_files = list(cache_path.glob("*.tflite"))
      if tflite_files:
        resolved_model_path = str(tflite_files[0])
      else:
        resolved_model_path = str(cache_path)
  except (json.JSONDecodeError, IOError):
    pass

  return resolved_model_path, resolved_hf_id
