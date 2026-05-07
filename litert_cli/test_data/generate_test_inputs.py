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

"""Helper script to generate dummy inputs for LiteRT CLI e2e tests."""

import sys
import numpy as np


def main(argv):
  if len(argv) < 2:
    print("Usage: generate_test_inputs.py <output_root>")
    sys.exit(1)
  if len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")
  output_root = argv[1]

  np.save(
      f"{output_root}/test_input.npy",
      np.full((1, 3, 224, 224), 0.5, dtype=np.float32),
  )

  np.full((1, 3, 224, 224), 0.5, dtype=np.float32).tofile(
      f"{output_root}/test_input.raw"
  )

  # Generate .png
  try:
    from PIL import Image  # pylint: disable=g-import-not-at-top

    img = Image.fromarray(np.full((224, 224, 3), 127, dtype=np.uint8))
    img.save(f"{output_root}/test_input.png")
  except ImportError:
    print("Pillow not found, skipping .png generation.")


if __name__ == "__main__":
  from absl import app  # pylint: disable=g-import-not-at-top
  app.run(main)
