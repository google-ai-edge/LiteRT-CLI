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

"""ResNet18 model for LiteRT conversion."""

import torch
import torchvision


def get_model(batch_size: int = 1) -> torch.nn.Module:
  model = torchvision.models.resnet18(
      weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
  )
  model.eval()
  return model


def get_args(batch_size: int = 1) -> tuple[torch.Tensor, ...]:
  return (torch.randn(batch_size, 3, 224, 224),)
