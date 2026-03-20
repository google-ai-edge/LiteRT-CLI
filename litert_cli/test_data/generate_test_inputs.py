"""Helper script to generate dummy inputs for LiteRT CLI e2e tests."""

import sys
import numpy as np
from PIL import Image


def main():
  if len(sys.argv) < 2:
    print("Usage: generate_test_inputs.py <output_root>")
    sys.exit(1)
  output_root = sys.argv[1]

  # Generate .npy
  np.save(
      f"{output_root}/test_input.npy",
      np.full((1, 3, 224, 224), 0.5, dtype=np.float32),
  )

  # Generate .raw
  np.full((1, 3, 224, 224), 0.5, dtype=np.float32).tofile(
      f"{output_root}/test_input.raw"
  )

  # Generate .png
  img = Image.fromarray(np.full((224, 224, 3), 127, dtype=np.uint8))
  img.save(f"{output_root}/test_input.png")


if __name__ == "__main__":
  main()
