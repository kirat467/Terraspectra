import pytest
import torch

from model_performance_report import generate_model_performance_report


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.layer(x)


def sample_evaluation():
    return {
        "accuracy": 0.9,
        "precision": 0.88,
        "recall": 0.9,
        "f1_score": 0.89,
        "mean_confidence": 0.95
    }


def sample_inference():
    return {
        "total_samples": 10,
        "total_batches": 5,
        "total_inference_time": 0.02,
        "average_batch_time": 0.004,
        "average_sample_time": 0.002,
        "samples_per_second": 500.0,
        "device": "cpu"
    }


def test_report_contains_model_name():
    result = generate_model_performance_report(
        SimpleModel(),
        sample_evaluation(),
        sample_inference(),
        model_name="Test CNN"
    )
    assert result["model_name"] == "Test CNN"


def test_report_contains_performance_metrics():
    result = generate_model_performance_report(
        SimpleModel(),
        sample_evaluation(),
        sample_inference()
    )
    assert result["performance"]["accuracy"] == 0.9
    assert result["performance"]["precision"] == 0.88
    assert result["performance"]["recall"] == 0.9
    assert result["performance"]["f1_score"] == 0.89


def test_report_contains_inference_metrics():
    result = generate_model_performance_report(
        SimpleModel(),
        sample_evaluation(),
        sample_inference()
    )
    assert result["inference"]["total_samples"] == 10
    assert result["inference"]["samples_per_second"] == 500.0


def test_report_contains_complexity_metrics():
    result = generate_model_performance_report(
        SimpleModel(),
        sample_evaluation(),
        sample_inference()
    )
    assert result["complexity"]["total_parameters"] == 10
    assert result["complexity"]["trainable_parameters"] == 10
    assert result["complexity"]["model_size_mb"] >= 0


def test_report_summary():
    result = generate_model_performance_report(
        SimpleModel(),
        sample_evaluation(),
        sample_inference()
    )
    assert result["summary"]["total_parameters"] == 10
    assert result["summary"]["evaluation_samples"] == 10
    assert result["summary"]["inference_speed"] == 500.0


def test_missing_evaluation_metric():
    evaluation = sample_evaluation()
    del evaluation["accuracy"]

    with pytest.raises(ValueError):
        generate_model_performance_report(
            SimpleModel(),
            evaluation,
            sample_inference()
        )


def test_invalid_model_name():
    with pytest.raises(ValueError):
        generate_model_performance_report(
            SimpleModel(),
            sample_evaluation(),
            sample_inference(),
            model_name=""
        )


def test_invalid_inference_results():
    with pytest.raises(ValueError):
        generate_model_performance_report(
            SimpleModel(),
            sample_evaluation(),
            []
        )