import pytest
from model_evaluation_summary import create_evaluation_summary


def test_valid_summary():
    metrics = {
        "accuracy": 0.9,
        "precision": 0.85,
        "recall": 0.8,
        "f1_score": 0.82,
        "test_samples": 100
    }
    result = create_evaluation_summary(metrics)
    assert result["accuracy"] == 0.9
    assert result["test_samples"] == 100


def test_rounding():
    metrics = {
        "accuracy": 0.87654,
        "precision": 0.81234,
        "recall": 0.76543,
        "f1_score": 0.78912,
        "test_samples": 50
    }
    result = create_evaluation_summary(metrics)
    assert result["accuracy"] == 0.8765


def test_missing_metric():
    with pytest.raises(ValueError):
        create_evaluation_summary({"accuracy": 0.9})


def test_invalid_input():
    with pytest.raises(TypeError):
        create_evaluation_summary([])


def test_invalid_metric_range():
    metrics = {
        "accuracy": 1.5,
        "precision": 0.8,
        "recall": 0.8,
        "f1_score": 0.8,
        "test_samples": 10
    }
    with pytest.raises(ValueError):
        create_evaluation_summary(metrics)


def test_negative_samples():
    metrics = {
        "accuracy": 0.9,
        "precision": 0.8,
        "recall": 0.8,
        "f1_score": 0.8,
        "test_samples": -1
    }
    with pytest.raises(ValueError):
        create_evaluation_summary(metrics)