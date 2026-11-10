
import torch
from model_3dcnn import Crop3DCNN

# Create sample hyperspectral images
images = torch.rand(4, 1, 200, 16, 16)

# Create model with 3 sample classes
model = Crop3DCNN(num_classes=3)

# Run model
outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)

# Check output
assert outputs.shape == (4, 3)

print("3D-CNN model test passed!")
