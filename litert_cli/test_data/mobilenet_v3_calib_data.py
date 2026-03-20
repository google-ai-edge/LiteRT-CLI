"""Calibration data for multiple vision models with dynamic input naming."""

import numpy as np


def get_calibration_data():
  dataset = []
  for _ in range(5):
    data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dataset.append({"args_0": data})

  return {"serving_default": dataset}
