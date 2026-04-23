"""Tests for inputs.py.

This module contains unit tests for verifying LiteRT CLI input parsing
operations (e.g., shape translation, array fallback).
"""

from __future__ import annotations

from collections.abc import Sequence
import pathlib
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from litert_cli.core import inputs
import numpy as np


class GetNpDtypeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="int32",
          tensor_type_str="int32",
          expected_dtype=np.int32,
      ),
      dict(
          testcase_name="INT32_upper",
          tensor_type_str="INT32",
          expected_dtype=np.int32,
      ),
      dict(
          testcase_name="int64",
          tensor_type_str="int64",
          expected_dtype=np.int64,
      ),
      dict(
          testcase_name="int8", tensor_type_str="int8", expected_dtype=np.int8
      ),
      dict(
          testcase_name="uint8",
          tensor_type_str="uint8",
          expected_dtype=np.uint8,
      ),
      dict(
          testcase_name="float16",
          tensor_type_str="float16",
          expected_dtype=np.float16,
      ),
      dict(
          testcase_name="fp16",
          tensor_type_str="fp16",
          expected_dtype=np.float16,
      ),
      dict(
          testcase_name="bool", tensor_type_str="bool", expected_dtype=np.bool_
      ),
      dict(
          testcase_name="float32",
          tensor_type_str="float32",
          expected_dtype=np.float32,
      ),
      dict(
          testcase_name="unknown",
          tensor_type_str="unknown",
          expected_dtype=np.float32,
      ),
  )
  def test_get_np_dtype(
      self, tensor_type_str: str, expected_dtype: np.dtype
  ) -> None:
    self.assertEqual(inputs.get_np_dtype(tensor_type_str), expected_dtype)


class ParseInputLiteralsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="single_float_scalar",
          data_string="1.5",
          shape=[1],
          dtype=np.float32,
          expected_arr=np.array([1.5], dtype=np.float32),
      ),
      dict(
          testcase_name="single_float_broadcast",
          data_string="1.5",
          shape=[2, 2],
          dtype=np.float32,
          expected_arr=np.array([[1.5, 1.5], [1.5, 1.5]], dtype=np.float32),
      ),
      dict(
          testcase_name="json_array_syntax",
          data_string="[1, 2, 3]",
          shape=[3],
          dtype=np.float32,
          expected_arr=np.array([1, 2, 3], dtype=np.float32),
      ),
      dict(
          testcase_name="multidimensional_array_syntax",
          data_string="[[1, 2], [3, 4]]",
          shape=[2, 2],
          dtype=np.float32,
          expected_arr=np.array([[1, 2], [3, 4]], dtype=np.float32),
      ),
      dict(
          testcase_name="comma_separated_fallback",
          data_string="1,2,3",
          shape=[3],
          dtype=np.float32,
          expected_arr=np.array([1, 2, 3], dtype=np.float32),
      ),
  )
  def test_parse_input_literals(
      self,
      data_string: str,
      shape: Sequence[int],
      dtype: np.dtype,
      expected_arr: np.ndarray,
  ) -> None:
    result = inputs.parse_input(data_string, shape, dtype)
    np.testing.assert_array_equal(result, expected_arr)


class ParseInputFilesTest(absltest.TestCase):

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
