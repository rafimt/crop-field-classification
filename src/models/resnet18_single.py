"""
Build a ResNet18 model that works with our satellite images.

Two changes to the standard ResNet18:
  1. The first layer normally expects 3 channels (red, green, blue). Our images
     have more channels (bands + NDVI + mask), so we widen it.
  2. The last layer normally predicts 1000 ImageNet classes. We swap it for one
     that predicts our crop classes.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_resnet18(num_input_channels, num_classes, pretrained=True):
    # Load ResNet18. With pretrained=True it comes with ImageNet weights,
    # which gives us a big head start (transfer learning).
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    # --- Change 1: widen the first convolution layer ---
    old_conv = model.conv1  # expects 3 input channels
    new_conv = nn.Conv2d(
        in_channels=num_input_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )

    # Fill the new layer's weights sensibly instead of starting from scratch.
    with torch.no_grad():
        if pretrained:
            old_weight = old_conv.weight               # shape (64, 3, 7, 7)
            average = old_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)

            # Start every input channel from the average of the RGB filters.
            new_weight = average.repeat(1, num_input_channels, 1, 1)

            # Keep the real RGB filters for the first 3 channels (if we have 3+).
            if num_input_channels >= 3:
                new_weight[:, :3] = old_weight

            new_conv.weight.copy_(new_weight)

    model.conv1 = new_conv

    # --- Change 2: replace the final layer with one for our classes ---
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
