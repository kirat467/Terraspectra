import torch
import pytest

from model_vit import CropViT
from predict import predict_crop


def create_model():
    return CropViT(input_dim=200, num_classes=3)


def test_predict_vit():
    class_names = ["Healthy", "Leaf Rust", "Blight"]

    model = create_model()
    images = torch.rand(4, 50, 200)

    results = predict_crop(model, images, class_names)

    assert len(results) == 4
    assert results[0]["prediction"] in class_names
    assert 0 <= results[0]["confidence"] <= 100


def test_vit_invalid_input_dimensions():
    model = create_model()
    images = torch.rand(4, 200)

    with pytest.raises(ValueError, match="Input image must be a 5D tensor"):
        predict_crop(model, images, ["Healthy", "Leaf Rust", "Blight"])


def test_vit_empty_batch():
    model = create_model()
    images = torch.rand(0, 50, 200)

    with pytest.raises(ValueError, match="Prediction batch cannot be empty"):
        predict_crop(model, images, ["Healthy", "Leaf Rust", "Blight"])


def test_vit_empty_sequence_dimension():
    model = create_model()
    images = torch.rand(4, 0, 200)

    with pytest.raises(ValueError, match="dimensions cannot be empty"):
        predict_crop(model, images, ["Healthy", "Leaf Rust", "Blight"])


def test_vit_empty_feature_dimension():
    model = create_model()
    images = torch.rand(4, 50, 0)

    with pytest.raises(ValueError, match="dimensions cannot be empty"):
        predict_crop(model, images, ["Healthy", "Leaf Rust", "Blight"])


def test_vit_nan_input_is_rejected():
    model = create_model()
    images = torch.rand(4, 50, 200)
    images[0, 0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite values"):
        predict_crop(model, images, ["Healthy", "Leaf Rust", "Blight"])


def test_vit_infinite_input_is_rejected():
    model = create_model()
    images = torch.rand(4, 50, 200)
    images[0, 0, 0] = float("inf")

    with pytest.raises(ValueError, match="finite values"):
        predict_crop(model, images, ["Healthy", "Leaf Rust", "Blight"])