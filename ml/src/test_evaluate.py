import torch
from torch.utils.data import TensorDataset, DataLoader

from model_3dcnn import Crop3DCNN
from evaluate import evaluate_model


class FixedOutputModel(torch.nn.Module):

    def forward(self, images):
        logits = torch.tensor(
            [5.0, 4.0, 3.0, 2.0, 1.0],
            device=images.device
        )
        return logits.unsqueeze(0).expand(images.size(0), -1)


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
    assert 0 <= results["top3_accuracy"] <= 1
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["f1_score"] <= 1

    assert results["class_labels"] == [0, 1, 2, 3, 4]

    assert results["class_names"] == [
        "Healthy",
        "Alternaria alternata",
        "Alternaria solani",
        "Botrytis cinerea",
        "Fusarium oxysporum"
    ]

    matrix = results["confusion_matrix"]

    assert len(matrix) == 5
    assert all(len(row) == 5 for row in matrix)
    assert sum(sum(row) for row in matrix) == 10

    counts = results["prediction_counts"]

    assert set(counts.keys()) == {0, 1, 2, 3, 4}
    assert sum(counts.values()) == 10

    report = results["classification_report"]

    for name in results["class_names"]:
        assert name in report

    confidence = results["mean_confidence"]

    assert 0 <= confidence <= 1


def test_top3_accuracy():

    images = torch.rand(2, 1)
    labels = torch.tensor([2, 3])

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=2)

    model = FixedOutputModel()

    results = evaluate_model(model, loader)

    # Class 2 is in top 3; class 3 is not.
    assert results["top3_accuracy"] == 0.5


def test_top3_accuracy_with_all_correct():

    images = torch.rand(3, 1)
    labels = torch.tensor([0, 1, 2])

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=3)

    model = FixedOutputModel()

    results = evaluate_model(model, loader)

    assert results["top3_accuracy"] == 1.0