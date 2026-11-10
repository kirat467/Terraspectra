
import torch
from torch.utils.data import TensorDataset
from model_3dcnn import Crop3DCNN
from train_3dcnn import train_model

# Create sample images and labels
images = torch.rand(8, 1, 200, 16, 16)
labels = torch.randint(0, 3, (8,))

dataset = TensorDataset(images, labels)

# Create model
model = Crop3DCNN(num_classes=3)

# Train for 2 epochs
trained_model = train_model(
    model,
    dataset,
    epochs=2,
    batch_size=2
)

print("3D-CNN training test passed!")
