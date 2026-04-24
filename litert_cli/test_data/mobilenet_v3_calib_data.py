"""Sample calibration data for MobileNetV3."""

import numpy as np


def get_calibration_data():
  dataset = [
      {"args_0": np.random.rand(1, 224, 224, 3).astype(np.float32)}
      for _ in range(5)
  ]
  return {"serving_default": dataset}
