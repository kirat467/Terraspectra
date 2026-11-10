
import torch
from model_vit import CropViT

# Create sample hyperspectral features
images = torch.rand(4, 50, 200)

# Create ViT model
model = CropViT(
    input_dim=200,
    num_classes=3
)

# Run prediction
outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)

assert outputs.shape == (4, 3)

print("ViT model test passed!")
