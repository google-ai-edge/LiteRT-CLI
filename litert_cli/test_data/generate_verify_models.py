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

r"""Generates the tiny models used by the `litert verify` tests.

Two models share one graph, `scaled = x * 2 + 1` and `total = sum(x)`, over a
`float32[1, 4]` input named `x`:

* `verify_float.tflite`: float32 inputs and outputs. `total` is a rank-0
  (scalar) output, which exercises the shape handling of `verify`.
* `verify_int8.tflite`: the same graph fully quantized with int8 inputs and
  outputs, used to check that `verify` refuses to compare models whose input
  dtypes differ instead of silently casting.

Usage:
  blaze run //third_party/py/litert_cli:generate_verify_models -- \
      --output_dir=/tmp/verify_models
  cp /tmp/verify_models/*.tflite third_party/py/litert_cli/test_data/
"""

from collections.abc import Sequence
import pathlib

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to write the .tflite files to.",
    required=True,
)

INPUT_SHAPE = (1, 4)


class _Model(tf.Module):
  """`scaled = x * 2 + 1`, `total = sum(x)`."""

  @tf.function(
      input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32, name="x")]
  )
  def __call__(self, x):
    return {
        "scaled": x * 2.0 + 1.0,
        "total": tf.reduce_sum(x),
    }


def _converter(model: _Model) -> tf.lite.TFLiteConverter:
  return tf.lite.TFLiteConverter.from_concrete_functions(
      [model.__call__.get_concrete_function()], model
  )


def _representative_dataset():
  rng = np.random.default_rng(0)
  for _ in range(64):
    yield [rng.uniform(-1.0, 1.0, size=INPUT_SHAPE).astype(np.float32)]


def main(argv: Sequence[str]) -> None:
  del argv
  out = pathlib.Path(_OUTPUT_DIR.value)
  out.mkdir(parents=True, exist_ok=True)
  model = _Model()

  (out / "verify_float.tflite").write_bytes(_converter(model).convert())

  converter = _converter(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = _representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  (out / "verify_int8.tflite").write_bytes(converter.convert())

  for path in sorted(out.glob("verify_*.tflite")):
    print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
  app.run(main)
