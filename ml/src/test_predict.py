
import torch

from model_3dcnn import Crop3DCNN
from predict import predict_crop


def test_predict_3dcnn():
    class_names = ["Healthy", "Leaf Rust", "Blight"]

    model = Crop3DCNN(num_classes=3)
    images = torch.rand(4, 1, 200, 16, 16)

    results = predict_crop(model, images, class_names)

    assert len(results) == 4
    assert results[0]["prediction"] in class_names
    assert 0 <= results[0]["confidence"] <= 100
