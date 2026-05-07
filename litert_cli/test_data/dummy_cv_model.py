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
