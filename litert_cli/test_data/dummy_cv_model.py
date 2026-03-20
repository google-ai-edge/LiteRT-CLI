import torch
import torch.nn as nn


class DummyVisionModel(nn.Module):

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


def get_model():
  return DummyVisionModel()


def get_args():
  # Return sample args as a tuple
  return (torch.randn(1, 3, 224, 224),)
