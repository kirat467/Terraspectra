
import torch
from torch.utils.data import TensorDataset, DataLoader

from model_vit import CropViT
from evaluate import evaluate_model


def test_evaluate_vit():
    images = torch.rand(12, 50, 200)
    labels = torch.randint(0, 3, (12,))

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4)

    model = CropViT(input_dim=200, num_classes=3)

    results = evaluate_model(model, loader)

    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1_score"] <= 1
