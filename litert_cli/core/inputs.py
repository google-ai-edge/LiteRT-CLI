"""Generic input parsing and preprocessing utilities.

This module provides utilities to parse input data for LiteRT CLI. For example,
input as an image, or a numpy array.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import pathlib
import types

from litert_cli.core.deps import ensure_extra
import numpy as np

_TENSOR_TYPE_TO_NP_DTYPE = types.MappingProxyType({
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "int8": np.int8,
    "float16": np.float16,
    "fp16": np.float16,
    "bool": np.bool_,
    "float32": np.float32,
})


def get_np_dtype(tensor_type_str: str) -> np.dtype:
  """Maps a LiteRT tensor type string to a numpy dtype.

  Args:
    tensor_type_str: A string describing the tensor type (e.g., "INT32"). If the
      type is unknown, falls back to `np.float32`.

  Returns:
    The corresponding numpy dtype (e.g., np.int32, np.float32).
  """
  return _TENSOR_TYPE_TO_NP_DTYPE.get(tensor_type_str.lower(), np.float32)


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

  if not ensure_extra("image", silent=True):
    raise ImportError(
        "Pillow (PIL) is required to process image inputs. Please run `pip"
        " install Pillow`."
    )
  from PIL import Image  # pylint: disable=g-import-not-at-top

  with Image.open(image_path) as img_file:
    img = img_file.convert("RGB")

  # Infer spatial dimensions (height, width) from the shape.
  # We assume the two largest dimensions greater than 1 are the spatial
  # dimensions (Height and Width). This handles common layouts like NHWC
  # ([B, H, W, C]) and NCHW ([B, C, H, W]), even with batch sizes > 1 or
  # non-standard channel counts, as long as H and W are the dominant sizes.
  potential_spatial_dims = sorted((s for s in shape if s > 1), reverse=True)

  if len(potential_spatial_dims) >= 2:
    height, width = potential_spatial_dims[:2]
    target_size = (width, height)  # (width, height) for PIL resize
    img = img.resize(target_size, Image.Resampling.BILINEAR)
  else:
    # Fallback to a common default if spatial dimensions aren't clearly found.
    img = img.resize((224, 224), Image.Resampling.BILINEAR)

  img_data = np.array(img)

  # PIL gives image data in HWC format. Check if the model expects NCHW.
  # A common NCHW shape has dimensions like [1, 3, H, W].
  if len(shape) == 4 and shape[1] == 3 and img_data.shape[2] == 3:
    # Transpose from HWC (height, width, channels) to
    # CHW (channels, height, width).
    img_data = img_data.transpose(2, 0, 1)

  # Reshape to expected model shape if possible (extremely basic matching)
  try:
    reshaped_img = img_data.reshape(shape)
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
    # TODO: Add support for custom normalization ranges (e.g., [-1, 1]),
    # as many vision models are trained with different ranges.
    processed_img = reshaped_img.astype(dtype) / 255.0
  else:
    # NOTE: For fully quantized models, this is a straight cast. Accurate
    # normalization requires querying model Scale/ZeroPoint parameters,
    # which is not available in this standalone context.
    processed_img = reshaped_img.astype(dtype)

  return processed_img


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
  try:
    if not data_path.exists():
      raise FileNotFoundError(f"File not found: {data_string}")

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
    elif ext in (".raw", ".bin"):
      data = np.fromfile(data_path, dtype=dtype)
      if data.size != np.prod(shape):
        raise ValueError(
            f"File size mismatch: expected {np.prod(shape)} elements, got"
            f" {data.size}"
        )
      return data.reshape(shape)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
      return _preprocess_image(data_path, shape, dtype)
    else:
      raise ValueError(
          f"Unsupported input file extension '{ext}' in basic parser."
      )
  except FileNotFoundError:
    # If not a file path, try parsing as a literal.
    # Try parsing as JSON array/literal (e.g., "[1, 2, 3]", "1.0", "[[1], [2]]")
    try:
      loaded = json.loads(data_string)
      arr = np.array(loaded, dtype=dtype)
      return np.broadcast_to(arr, shape).astype(dtype)
    except (json.JSONDecodeError, ValueError):
      pass

    # Fallback for "1,2,3" without brackets
    try:
      arr = np.fromstring(data_string, sep=",", dtype=dtype)
      return np.broadcast_to(arr, shape).astype(dtype)
    except ValueError:
      pass

  raise ValueError(f"Could not parse input data: {data_string!r}")
