"""Dummy vision model for testing."""

import torch
from torch import nn


class DummyVisionModel(nn.Module):
  """A dummy vision model with a conv layer, relu, pool, and fc layer.

  Attributes:
    conv: The convolutional layer.
    relu: The ReLU activation layer.
    pool: The adaptive average pooling layer.
    fc: The fully connected layer.
  """

  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3, 16, 3)
    self.relu = nn.ReLU()
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(16, 10)

  def forward(self, x):
    x = self.conv(x)
    x = self.relu(x)
    x = self.pool(x)
    x = torch.flatten(x, 1)
    return self.fc(x)


def get_model() -> DummyVisionModel:
  return DummyVisionModel()


def get_args() -> tuple[torch.Tensor, ...]:
  return (torch.randn(1, 3, 224, 224),)
