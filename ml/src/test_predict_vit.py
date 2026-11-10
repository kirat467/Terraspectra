
import torch
from model_vit import CropViT
from predict import predict_crop

# Sample class names
class_names = ["Healthy", "Leaf Rust", "Blight"]

# Create ViT model
model = CropViT(input_dim=200, num_classes=3)

# Create sample spectral features
images = torch.rand(4, 50, 200)

# Get predictions
results = predict_crop(model, images, class_names)

for i, result in enumerate(results):
    print(f"Image {i+1}:")
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"], "%")

print("\nViT prediction test passed!")
