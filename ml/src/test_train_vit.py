import torch
import pytest

from torch.utils.data import TensorDataset

from model_vit import CropViT
from train_vit import train_vit, save_vit


def create_model():
    return CropViT(
        input_dim=200,
        num_classes=3
    )


def create_dataset():
    images = torch.rand(12, 50, 200)
    labels = torch.randint(0, 3, (12,))

    return TensorDataset(images, labels)


def test_train_vit():

    dataset = create_dataset()
    model = create_model()

    trained_model = train_vit(
        model,
        dataset,
        epochs=2,
        batch_size=4
    )

    assert isinstance(trained_model, CropViT)


def test_save_vit(tmp_path):

    model = create_model()
    save_path = tmp_path / "vit.pth"

    save_vit(model, save_path)

    assert save_path.exists()


def test_empty_training_dataset():

    images = torch.empty(0, 50, 200)
    labels = torch.empty(0, dtype=torch.long)

    dataset = TensorDataset(images, labels)

    with pytest.raises(ValueError, match="dataset is empty"):
        train_vit(create_model(), dataset)


def test_invalid_epochs():

    with pytest.raises(ValueError, match="Invalid training parameters"):
        train_vit(create_model(), create_dataset(), epochs=0)


def test_invalid_batch_size():

    with pytest.raises(ValueError, match="Invalid training parameters"):
        train_vit(create_model(), create_dataset(), batch_size=0)


def test_invalid_learning_rate():

    with pytest.raises(ValueError, match="Invalid training parameters"):
        train_vit(create_model(), create_dataset(), lr=0)


def test_invalid_training_image_dimensions():

    images = torch.rand(8, 200)
    labels = torch.randint(0, 3, (8,))

    dataset = TensorDataset(images, labels)

    with pytest.raises(ValueError, match="must be 3D"):
        train_vit(create_model(), dataset)


def test_invalid_training_labels():

    images = torch.rand(8, 50, 200)
    labels = torch.randint(0, 3, (8, 1))

    dataset = TensorDataset(images, labels)

    with pytest.raises(ValueError, match="labels must be 1D"):
        train_vit(create_model(), dataset)


def test_nan_training_input():

    images = torch.rand(8, 50, 200)
    images[0, 0, 0] = float("nan")

    labels = torch.randint(0, 3, (8,))

    dataset = TensorDataset(images, labels)

    with pytest.raises(ValueError, match="finite values"):
        train_vit(create_model(), dataset)


def test_invalid_class_labels():

    images = torch.rand(8, 50, 200)
    labels = torch.full((8,), 5, dtype=torch.long)

    dataset = TensorDataset(images, labels)

    with pytest.raises(ValueError, match="invalid class indices"):
        train_vit(create_model(), dataset)