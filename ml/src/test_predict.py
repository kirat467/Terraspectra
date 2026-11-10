
import torch
from model_3dcnn import Crop3DCNN
from predict import predict_crop

# Sample class names
class_names = ["Healthy", "Leaf Rust", "Blight"]

# Create model
model = Crop3DCNN(num_classes=3)

# Create 4 sample hyperspectral images
images = torch.rand(4, 1, 200, 16, 16)

# Get predictions
results = predict_crop(model, images, class_names)

for i, result in enumerate(results):
    print(f"Image {i+1}:")
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"], "%")

print("\nPrediction test passed!")
