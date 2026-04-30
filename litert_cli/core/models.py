"""Model plugin for LiteRT CLI."""

from __future__ import annotations

import json
import pathlib

from litert_cli.core import constants


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

  ref_name = main_ref.replace("/", "__") if "/" in main_ref else main_ref
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
