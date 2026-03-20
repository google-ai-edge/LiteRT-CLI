"""Tests for inputs.py.

This module contains unit tests for verifying LiteRT CLI input parsing
operations (e.g., shape translation, array fallback).
"""

import pathlib
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from litert_cli.core import inputs
import numpy as np


class GetNpDtypeTest(parameterized.TestCase):
  """Tests for get_np_dtype."""

  @parameterized.parameters(
      ("int32", np.int32),
      ("INT32", np.int32),
      ("int64", np.int64),
      ("int8", np.int8),
      ("uint8", np.uint8),
      ("float16", np.float16),
      ("fp16", np.float16),
      ("bool", np.bool_),
      ("float32", np.float32),
      ("unknown", np.float32),  # Default fallback
  )
  def test_get_np_dtype(
      self, tensor_type_str: str, expected_dtype: np.dtype
  ) -> None:
    self.assertEqual(inputs.get_np_dtype(tensor_type_str), expected_dtype)


class ParseInputLiteralsTest(parameterized.TestCase):
  """Tests for parse_input with literal strings."""

  @parameterized.parameters(
      # Single float scalar
      ("1.5", [1], np.float32, np.array([1.5], dtype=np.float32)),
      # Single float broadcast
      (
          "1.5",
          [2, 2],
          np.float32,
          np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float32),
      ),
      # JSON array syntax
      ("[1, 2, 3]", [3], np.float32, np.array([1, 2, 3], dtype=np.float32)),
      # Multidimensional array syntax
      (
          "[[1, 2], [3, 4]]",
          [2, 2],
          np.float32,
          np.array([[1, 2], [3, 4]], dtype=np.float32),
      ),
      # Comma separated fallback without brackets
      ("1,2,3", [3], np.float32, np.array([1, 2, 3], dtype=np.float32)),
  )
  def test_parse_input_literals(
      self,
      data_string: str,
      shape: list[int],
      dtype: np.dtype,
      expected_arr: np.ndarray,
  ) -> None:
    result = inputs.parse_input(data_string, shape, dtype)
    np.testing.assert_array_equal(result, expected_arr)


class ParseInputFilesTest(absltest.TestCase):
  """Tests for parse_input with file paths."""

  def setUp(self) -> None:
    super().setUp()
    self.test_dir = pathlib.Path(tempfile.mkdtemp())

  def tearDown(self) -> None:
    shutil.rmtree(self.test_dir)
    super().tearDown()

  def test_parse_npy_file(self) -> None:
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    filepath = self.test_dir / "test.npy"
    np.save(filepath, arr)

    result = inputs.parse_input(str(filepath), [2, 2], np.float32)
    np.testing.assert_array_equal(result, arr)

  def test_parse_raw_file(self) -> None:
    arr = np.array([1, 2, 3, 4], dtype=np.int32)
    filepath = self.test_dir / "test.raw"
    arr.tofile(filepath)

    result = inputs.parse_input(str(filepath), [4], np.int32)
    np.testing.assert_array_equal(result, arr)

  def test_parse_unsupported_file_extension_raises(self) -> None:
    filepath = self.test_dir / "test.txt"
    filepath.write_text("dummy")

    with self.assertRaisesRegex(ValueError, "Unsupported input file extension"):
      inputs.parse_input(str(filepath), [1], np.float32)


if __name__ == "__main__":
  absltest.main()
