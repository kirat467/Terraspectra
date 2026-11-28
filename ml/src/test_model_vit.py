import torch
import pytest

from model_vit import CropViT


def create_model():
    return CropViT(
        input_dim=200,
        num_classes=3
    )


def test_vit_output_shape():
    model = create_model()
    images = torch.rand(4, 50, 200)

    outputs = model(images)

    assert outputs.shape == (4, 3)


def test_vit_invalid_dimensions():
    model = create_model()
    images = torch.rand(4, 200)

    with pytest.raises(ValueError, match="ViT input must be 3D"):
        model(images)


def test_vit_empty_batch():
    model = create_model()
    images = torch.rand(0, 50, 200)

    with pytest.raises(ValueError, match="batch cannot be empty"):
        model(images)


def test_vit_empty_sequence():
    model = create_model()
    images = torch.rand(4, 0, 200)

    with pytest.raises(ValueError, match="Sequence length"):
        model(images)


def test_vit_invalid_feature_dimension():
    model = create_model()
    images = torch.rand(4, 50, 100)

    with pytest.raises(ValueError, match="input features"):
        model(images)


def test_vit_invalid_input_dimension_parameter():
    with pytest.raises(ValueError, match="Input dimension"):
        CropViT(input_dim=0, num_classes=3)


def test_vit_invalid_attention_heads():
    with pytest.raises(ValueError, match="divisible"):
        CropViT(
            input_dim=200,
            num_classes=3,
            embed_dim=127,
            num_heads=4
        )


def test_vit_invalid_number_of_classes():
    with pytest.raises(ValueError, match="Number of classes"):
        CropViT(input_dim=200, num_classes=0)