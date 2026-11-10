
import torch
from torch.utils.data import TensorDataset, DataLoader

from model_vit import CropViT
from evaluate import evaluate_model

# Generate sample data
images = torch.rand(12, 50, 200)
labels = torch.randint(0, 3, (12,))

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=4)

# Create ViT model
model = CropViT(input_dim=200, num_classes=3)

# Evaluate model
evaluate_model(model, loader)

print("\nViT evaluation test passed!")
