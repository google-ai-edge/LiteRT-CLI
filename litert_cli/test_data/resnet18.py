import torch
import torchvision


def get_model():
  model = torchvision.models.resnet18(
      weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
  )
  model.eval()
  return model


def get_args():
  return (torch.randn(1, 3, 224, 224),)
