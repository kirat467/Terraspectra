
import torch
from torch.utils.data import TensorDataset

from model_vit import CropViT
from train_vit import train_vit, save_vit


def test_train_vit():
    images = torch.rand(12, 50, 200)
    labels = torch.randint(0, 3, (12,))

    dataset = TensorDataset(images, labels)

    model = CropViT(input_dim=200, num_classes=3)

    trained_model = train_vit(
        model,
        dataset,
        epochs=2,
        batch_size=4
    )

    assert isinstance(trained_model, CropViT)


def test_save_vit(tmp_path):
    model = CropViT(input_dim=200, num_classes=3)

    save_path = tmp_path / "vit.pth"
    save_vit(model, save_path)

    assert save_path.exists()
