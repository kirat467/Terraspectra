import torch
import pytest

from model_3dcnn import Crop3DCNN
from predict import predict_crop


class FixedPredictionModel(torch.nn.Module):

    def forward(self, images):
        return torch.tensor(
            [[5.0, 3.0, 1.0]],
            device=images.device
        ).expand(images.shape[0], -1)


class InvalidPredictionModel(torch.nn.Module):

    def forward(self, images):
        return torch.tensor(
            [[float("nan"), 2.0, 1.0]],
            device=images.device
        ).expand(images.shape[0], -1)


def test_predict_3dcnn():

    class_names = ["Healthy", "Leaf Rust", "Blight"]

    model = Crop3DCNN(num_classes=3)
    images = torch.rand(4, 1, 200, 16, 16)

    results = predict_crop(model, images, class_names)

    assert len(results) == 4
    assert results[0]["prediction"] in class_names
    assert 0 <= results[0]["confidence"] <= 100


def test_top_predictions_are_sorted():

    images = torch.rand(2, 1)
    class_names = ["Healthy", "Rust", "Blight"]

    results = predict_crop(
        FixedPredictionModel(),
        images,
        class_names
    )

    predictions = results[0]["top_predictions"]

    assert len(predictions) == 3
    assert predictions[0]["disease"] == "Healthy"

    confidences = [
        item["confidence"] for item in predictions
    ]

    assert confidences == sorted(confidences, reverse=True)


def test_prediction_confidences_are_valid():

    images = torch.rand(1, 1)

    results = predict_crop(
        FixedPredictionModel(),
        images,
        ["Healthy", "Rust", "Blight"]
    )

    total = sum(
        item["confidence"]
        for item in results[0]["top_predictions"]
    )

    assert total == pytest.approx(100.0, abs=0.02)


def test_invalid_nan_outputs_are_rejected():

    images = torch.rand(1, 1)

    with pytest.raises(ValueError, match="finite"):
        predict_crop(
            InvalidPredictionModel(),
            images,
            ["Healthy", "Rust", "Blight"]
        )