import torch
from torch.utils.data import TensorDataset, DataLoader

from model_3dcnn import Crop3DCNN
from evaluate import evaluate_model


def test_evaluate_3dcnn():

    images = torch.rand(10, 1, 200, 16, 16)

    labels = torch.tensor([
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4
    ])

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2)

    model = Crop3DCNN(num_classes=5)

    results = evaluate_model(model, loader)

    assert 0 <= results["accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1_score"] <= 1

    # Check all five classes
    assert results["class_labels"] == [0, 1, 2, 3, 4]

    assert results["class_names"] == [
        "Healthy",
        "Alternaria alternata",
        "Alternaria solani",
        "Botrytis cinerea",
        "Fusarium oxysporum"
    ]

    # Check confusion matrix
    matrix = results["confusion_matrix"]

    assert len(matrix) == 5
    assert all(len(row) == 5 for row in matrix)
    assert sum(sum(row) for row in matrix) == 10

    # Check prediction distribution
    counts = results["prediction_counts"]

    assert set(counts.keys()) == {0, 1, 2, 3, 4}
    assert sum(counts.values()) == 10

    # Check classification report
    report = results["classification_report"]

    for name in results["class_names"]:
        assert name in report

    # Check prediction confidence
    confidence = results["mean_confidence"]

    assert 0 <= confidence <= 1