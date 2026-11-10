
import torch
from torch.utils.data import TensorDataset
from model_vit import CropViT
from train_vit import train_vit

# Sample features and labels
images = torch.rand(12, 50, 200)
labels = torch.randint(0, 3, (12,))

dataset = TensorDataset(images, labels)

# Create ViT model
model = CropViT(input_dim=200, num_classes=3)

# Train for 2 epochs
trained_model = train_vit(
    model,
    dataset,
    epochs=2,
    batch_size=4
)

print("ViT training test passed!")
