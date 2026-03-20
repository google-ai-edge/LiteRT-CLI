"""Generic input parsing and preprocessing utilities.

This module provides utilities to parse input data for LiteRT CLI. For example,
input as a image, or a numpy array.
"""

from __future__ import annotations

import json
import pathlib
from typing import Sequence

import numpy as np


def get_np_dtype(tensor_type_str: str) -> np.dtype:
  """Maps a LiteRT tensor type string to a numpy dtype.

  Args:
    tensor_type_str: A string describing the tensor type (e.g., "INT32").

  Returns:
    The corresponding numpy dtype (e.g., np.int32, np.float32).
  """
  tensor_type_lower = str(tensor_type_str).lower()
  if "int32" in tensor_type_lower:
    return np.int32
  elif "int64" in tensor_type_lower:
    return np.int64
  elif "uint8" in tensor_type_lower:
    return np.uint8
  elif "int8" in tensor_type_lower:
    return np.int8
  elif "float16" in tensor_type_lower or "fp16" in tensor_type_lower:
    return np.float16
  elif "bool" in tensor_type_lower:
    return np.bool_
  return np.float32


def _preprocess_image(
    image_path: str | pathlib.Path, shape: Sequence[int], dtype: np.dtype
) -> np.ndarray:
  """Loads and preprocesses an image to the target shape and dtype.

  Installs Pillow on-the-fly, if not installed yet.

  Args:
    image_path: Path to the image file.
    shape: Expected model input shape (e.g., [1, 224, 224, 3]).
    dtype: Target numpy dtype for the processed image.

  Returns:
    A preprocessed numpy array ready for model inference.

  Raises:
    ImportError: If Pillow (PIL) is not installed.
    ValueError: If image cannot be reshaped to the target shape.
  """

  from litert_cli.core.deps import ensure_extra  # pylint: disable=g-import-not-at-top

  if not ensure_extra("image", silent=True):
    raise ImportError(
        "Pillow (PIL) is required to process image inputs. Please run `pip"
        " install Pillow`."
    )
  from PIL import Image  # pylint: disable=g-import-not-at-top

  img = Image.open(image_path).convert("RGB")

  # Typical shapes are [1, 224, 224, 3] or [1, 3, 224, 224]
  # We try to infer spatial dimensions.
  # Heuristic: Assume dimensions > 4 are spatial (height/width) to filter out
  # batch index (1) or channels (1, 3, 4).
  # NOTE: Might fail for extremely low-resolution inputs (e.g., 2x2).
  spatial_shape = [s for s in shape if s > 4]
  if len(spatial_shape) >= 2:
    target_size = (
        spatial_shape[1],
        spatial_shape[0],
    )  # (width, height) for PIL resize
    img = img.resize(target_size, Image.Resampling.BILINEAR)
  else:
    # Fallback to a common default if shape isn't obvious
    img = img.resize((224, 224), Image.Resampling.BILINEAR)

  img_data = np.array(img)

  # Reshape to expected model shape if possible (extremely basic matching)
  try:
    img_data = img_data.reshape(shape)
  except ValueError as e:
    raise ValueError(
        f"Failed to reshape image to expected layout shape {shape}. Detected"
        f" layout: {img_data.shape}"
    ) from e

  if np.issubdtype(dtype, np.floating):
    # Standardize to [0, 1]. Users might need specific mean/std depending on
    # the exact model (e.g., MobileNetV2 uses [-1, 1]). For general safety
    # when model preprocessing is unknown, [0, 1] normalization is a safe
    # default fallback.
    img_data = img_data.astype(dtype) / 255.0
  else:
    # NOTE: For fully quantized models, this is a straight cast. Accurate
    # normalization requires querying model Scale/ZeroPoint parameters,
    # which is not available in this standalone context.
    img_data = img_data.astype(dtype)

  return img_data


def parse_input(
    data_string: str, shape: Sequence[int], dtype: np.dtype
) -> np.ndarray:
  """Parses a single input string into a numpy array.

  Supports parsing from files (.npy, .raw, .bin), image files (jpg, png) or
  literal array notation (e.g., "[1, 2, 3]", "1.0", "1,2,3").

  Args:
    data_string: The string representation of the input (path or literal).
    shape: Expected template shape for broadcasting or loading.
    dtype: Target numpy dtype for output array.

  Returns:
    A numpy array populated with input coefficients.

  Raises:
    ValueError: If folder directory input attempted or parsing triggers error.
  """
  data_path = pathlib.Path(data_string)
  if data_path.exists():
    if data_path.is_dir():
      # Try finding a matching file if directory provided...
      # Complicated without name. For now, we can't easily guess.
      raise ValueError(
          "Directory input not yet supported in this basic parser:"
          f" {data_string}"
      )

    ext = data_path.suffix.lower()
    if ext == ".npy":
      return np.load(data_path).astype(dtype).reshape(shape)
    elif ext in [".raw", ".bin"]:
      data = np.fromfile(data_path, dtype=dtype)
      return data.reshape(shape)
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
      return _preprocess_image(data_path, shape, dtype)
    else:
      raise ValueError(
          f"Unsupported input file extension '{ext}' in basic parser."
      )

  # Try parsing as JSON array/literal (e.g., "[1, 2, 3]", "1.0", "[[1], [2]]")
  try:
    loaded = json.loads(data_string)
    arr = np.array(loaded, dtype=dtype)
    return np.broadcast_to(arr, shape).astype(dtype)
  except (json.JSONDecodeError, ValueError):
    pass

  # Fallback for "1,2,3" without brackets
  try:
    arr = np.array([float(x) for x in data_string.split(",")], dtype=dtype)
    return np.broadcast_to(arr, shape).astype(dtype)
  except ValueError:
    pass

  raise ValueError(f"Could not parse input data: {data_string}")
