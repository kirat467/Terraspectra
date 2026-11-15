import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from inference_analysis import analyze_inference_time


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.layer(x)


def create_loader():
    images = torch.rand(10, 4)
    labels = torch.randint(0, 2, (10,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=2)


def test_inference_analysis():
    model = SimpleModel()
    loader = create_loader()

    result = analyze_inference_time(
        model, loader, device=torch.device("cpu")
    )

    assert result["total_samples"] == 10
    assert result["total_batches"] == 5
    assert result["total_inference_time"] >= 0
    assert result["average_batch_time"] >= 0
    assert result["average_sample_time"] >= 0
    assert result["device"] == "cpu"


def test_samples_per_second():
    result = analyze_inference_time(
        SimpleModel(),
        create_loader(),
        device=torch.device("cpu")
    )
    assert result["samples_per_second"] > 0


def test_empty_dataset():
    images = torch.empty(0, 4)
    labels = torch.empty(0, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels))

    with pytest.raises(ValueError):
        analyze_inference_time(
            SimpleModel(), loader, device=torch.device("cpu")
        )


def test_invalid_warmup():
    with pytest.raises(ValueError):
        analyze_inference_time(
            SimpleModel(),
            create_loader(),
            device=torch.device("cpu"),
            warmup=-1
        )


def test_model_evaluation_mode():
    model = SimpleModel()
    model.train()

    analyze_inference_time(
        model, create_loader(), device=torch.device("cpu")
    )

    assert not model.training