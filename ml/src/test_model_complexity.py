import torch

from model_complexity import analyze_model_complexity


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(4, 8)
        self.layer2 = torch.nn.Linear(8, 2)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


def test_total_parameters():
    result = analyze_model_complexity(SimpleModel())
    assert result["total_parameters"] == 58


def test_trainable_parameters():
    result = analyze_model_complexity(SimpleModel())
    assert result["trainable_parameters"] == 58


def test_non_trainable_parameters():
    result = analyze_model_complexity(SimpleModel())
    assert result["non_trainable_parameters"] == 0


def test_total_layers():
    result = analyze_model_complexity(SimpleModel())
    assert result["total_layers"] == 2


def test_model_size():
    result = analyze_model_complexity(SimpleModel())
    assert result["model_size_mb"] > 0


def test_frozen_parameters():
    model = SimpleModel()
    model.layer1.weight.requires_grad = False

    result = analyze_model_complexity(model)

    assert result["trainable_parameters"] == 26
    assert result["non_trainable_parameters"] == 32