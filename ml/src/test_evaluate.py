
import torch
from torch.utils.data import TensorDataset, DataLoader

from model_3dcnn import Crop3DCNN
from evaluate import evaluate_model

# Generate sample data
images = torch.rand(12, 1, 200, 16, 16)
labels = torch.randint(0, 3, (12,))

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=4)

# Create model
model = Crop3DCNN(num_classes=3)

# Evaluate model
evaluate_model(model, loader)

print("\n3D-CNN evaluation test passed!")
