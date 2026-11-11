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

"""Tests for the shared desktop inference helpers."""

from __future__ import annotations

import os

from absl.testing import absltest
import click
from click import testing
from litert_cli import litert
from litert_cli.core import runtime
import numpy as np

_TEST_DATA = os.path.join(os.path.dirname(__file__), '..', 'test_data')
# `scaled = x * 2 + 1` (float32[1, 4]) and `total = sum(x)` (scalar) over a
# float32 input `x`; and the same graph fully quantized with int8 I/O.
_FLOAT_MODEL = os.path.join(_TEST_DATA, 'verify_float.tflite')
_INT8_MODEL = os.path.join(_TEST_DATA, 'verify_int8.tflite')


class ParseAcceleratorsTest(absltest.TestCase):

  def test_defaults_to_cpu(self) -> None:
    self.assertEqual(runtime.parse_accelerators(None), ['cpu'])
    self.assertEqual(runtime.parse_accelerators(''), ['cpu'])
    self.assertEqual(runtime.parse_accelerators(' , '), ['cpu'])

  def test_flags_are_ordered_npu_gpu_cpu(self) -> None:
    self.assertEqual(runtime.parse_accelerators(gpu=True), ['gpu'])
    self.assertEqual(
        runtime.parse_accelerators(cpu=True, gpu=True, npu=True),
        ['npu', 'gpu', 'cpu'],
    )

  def test_spec_wins_over_flags_and_is_normalized(self) -> None:
    self.assertEqual(
        runtime.parse_accelerators(' GPU, cpu ', npu=True), ['gpu', 'cpu']
    )

  def test_unknown_names_are_rejected(self) -> None:
    with self.assertRaisesRegex(click.ClickException, "'tpu'"):
      runtime.parse_accelerators('gpu,tpu')


class DtypeHelpersTest(absltest.TestCase):

  def test_np_dtype_rejects_types_buffers_cannot_move(self) -> None:
    self.assertEqual(runtime.np_dtype('float32', tensor='x'), np.float32)
    self.assertEqual(runtime.np_dtype('INT8', tensor='x'), np.int8)
    self.assertEqual(runtime.np_dtype('bool', tensor='x'), np.bool_)
    for unsupported in ('float64', 'int16', 'uint16', 'uint32', 'unknown'):
      with self.subTest(unsupported):
        with self.assertRaisesRegex(click.ClickException, unsupported):
          runtime.np_dtype(unsupported, tensor='x')

  def test_random_input_matches_dtype(self) -> None:
    rng = np.random.default_rng(0)
    bools = runtime.random_input(rng, [2, 64], np.dtype(np.bool_))
    self.assertEqual(bools.dtype, np.bool_)
    self.assertTrue(bools.any() and not bools.all())  # not all True
    ints = runtime.random_input(rng, [8], np.dtype(np.int8))
    self.assertEqual(ints.dtype, np.int8)
    self.assertTrue(((ints >= 0) & (ints < 10)).all())
    floats = runtime.random_input(rng, [8], np.dtype(np.float16))
    self.assertEqual(floats.dtype, np.float16)
    self.assertTrue(((floats >= -1.0) & (floats < 1.0)).all())

  def test_coerce_input_casts_floats_but_not_across_kinds(self) -> None:
    f32 = np.dtype(np.float32)
    out = runtime.coerce_input('x', np.array([[1.0, 2.0]]), [1, 2], f32)
    self.assertEqual(out.dtype, np.float32)
    with self.assertRaisesRegex(click.ClickException, 'expects int8'):
      runtime.coerce_input(
          'x', np.zeros((1, 2), np.float32), [1, 2], np.dtype(np.int8)
      )
    with self.assertRaisesRegex(click.ClickException, 'expects float32'):
      runtime.coerce_input('x', np.zeros((1, 2), np.int8), [1, 2], f32)
    with self.assertRaisesRegex(click.ClickException, 'has shape'):
      runtime.coerce_input('x', np.zeros((2,), np.float32), [1, 2], f32)


class RunOnceTest(absltest.TestCase):

  def test_runs_given_inputs_and_keeps_scalar_outputs_rank_0(self) -> None:
    x = np.array([[0.5, -1.0, 2.0, 0.25]])  # float64, as numpy makes them
    sig_key, fed, outputs = runtime.run_once(_FLOAT_MODEL, ['cpu'], 0, {'x': x})
    self.assertEqual(sig_key, 'serving_default')
    self.assertEqual(fed['x'].dtype, np.float32)
    np.testing.assert_allclose(outputs['scaled'], x * 2 + 1, rtol=1e-6)
    self.assertEqual(outputs['total'].shape, ())
    self.assertEqual(outputs['total'].dtype, np.float32)
    self.assertAlmostEqual(float(outputs['total']), 1.75, places=5)

  def test_random_inputs_are_seeded_and_match_the_signature(self) -> None:
    _, fed_a, out_a = runtime.run_once(_FLOAT_MODEL, ['cpu'], 0, None, seed=7)
    _, fed_b, out_b = runtime.run_once(_FLOAT_MODEL, ['cpu'], 0, None, seed=7)
    _, fed_c, _ = runtime.run_once(_FLOAT_MODEL, ['cpu'], 0, None, seed=8)
    self.assertEqual(fed_a['x'].shape, (1, 4))
    self.assertEqual(fed_a['x'].dtype, np.float32)
    np.testing.assert_array_equal(fed_a['x'], fed_b['x'])
    np.testing.assert_array_equal(out_a['scaled'], out_b['scaled'])
    self.assertFalse(np.array_equal(fed_a['x'], fed_c['x']))

  def test_int8_model_gets_int8_inputs_and_outputs(self) -> None:
    _, fed, outputs = runtime.run_once(_INT8_MODEL, ['cpu'], 0, None)
    self.assertEqual(fed['x'].dtype, np.int8)
    self.assertEqual({o.dtype for o in outputs.values()}, {np.dtype(np.int8)})

  def test_float_inputs_to_an_int8_model_are_rejected(self) -> None:
    with self.assertRaisesRegex(click.ClickException, 'expects int8'):
      runtime.run_once(
          _INT8_MODEL, ['cpu'], 0, {'x': np.zeros((1, 4), np.float32)}
      )

  def test_missing_and_out_of_range_signature(self) -> None:
    with self.assertRaisesRegex(click.ClickException, 'out of range'):
      runtime.run_once(_FLOAT_MODEL, ['cpu'], 3, None)
    with self.assertRaisesRegex(click.ClickException, "no entry for.*'x'"):
      runtime.run_once(_FLOAT_MODEL, ['cpu'], 0, {'y': np.zeros((1, 4))})


class RunDesktopUsesRuntimeTest(absltest.TestCase):
  """`litert run --desktop` goes through the same helpers."""

  def test_run_prints_a_scalar_output(self) -> None:
    result = testing.CliRunner().invoke(
        litert.cli,
        [
            'run',
            _FLOAT_MODEL,
            '--desktop',
            '--print-tensors',
            '--input',
            'x=[[1,2,3,4]]',
        ],
    )
    self.assertEqual(result.exit_code, 0, result.output)
    self.assertIn('accelerators: cpu', result.output)
    self.assertIn('scaled (shape: [1, 4])', result.output)
    self.assertIn('[3. 5. 7. 9.]', result.output)
    self.assertIn('total (shape: [])', result.output)
    self.assertIn('10.', result.output)

  def test_run_rejects_an_unknown_accelerator_before_loading(self) -> None:
    result = testing.CliRunner().invoke(
        litert.cli, ['run', _FLOAT_MODEL, '--desktop', '--accelerator', 'tpu']
    )
    self.assertEqual(result.exit_code, 1, result.output)
    self.assertIn("Unsupported hardware accelerator: 'tpu'", result.output)
    self.assertNotIn('Loading model', result.output)


if __name__ == '__main__':
  absltest.main()
