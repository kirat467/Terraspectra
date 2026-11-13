import torch
from torch.utils.data import TensorDataset, DataLoader

from model_3dcnn import Crop3DCNN
from evaluate import evaluate_model


def test_evaluate_3dcnn():
    images = torch.rand(12, 1, 200, 16, 16)
    labels = torch.randint(0, 3, (12,))

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=4)

    model = Crop3DCNN(num_classes=3)
    results = evaluate_model(model, loader)

    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1_score"] <= 1

    matrix = results["confusion_matrix"]
    assert len(matrix) == len(results["class_labels"])
    assert sum(sum(row) for row in matrix) == 12

    counts = results["prediction_counts"]
    assert sum(counts.values()) == 12
    assert set(counts.keys()) == {0, 1, 2}

    confidence = results["mean_confidence"]
    assert 0 <= confidence <= 1