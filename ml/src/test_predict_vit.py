
import torch

from model_vit import CropViT
from predict import predict_crop


def test_predict_vit():
    class_names = ["Healthy", "Leaf Rust", "Blight"]

    model = CropViT(input_dim=200, num_classes=3)
    images = torch.rand(4, 50, 200)

    results = predict_crop(model, images, class_names)

    assert len(results) == 4
    assert results[0]["prediction"] in class_names
    assert 0 <= results[0]["confidence"] <= 100
