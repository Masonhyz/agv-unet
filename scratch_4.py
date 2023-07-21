import torch
import torch.nn as nn


# Create a tensor of shape (batch size, 1, 32, 32)
tensor_2d = torch.randn(1, 1, 32, 32)
up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

upsampled_bilinear = up2(tensor_2d)
upsampled_trilinear = up3(tensor_2d)
# Upsample using trilinear interpolation

# Print the shapes of the original and upsampled tensors
print("Original tensor shape:", tensor_2d.shape)
print("Bilinear upsampled tensor shape:", upsampled_bilinear.shape)
print("Trilinear upsampled tensor shape:", upsampled_trilinear.shape)
print(upsampled_trilinear == upsampled_bilinear)


