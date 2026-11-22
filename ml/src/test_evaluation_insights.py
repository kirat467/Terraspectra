import pytest
from evaluation_insights import generate_evaluation_insights


def test_strong_accuracy():
    metrics = {"accuracy": 0.9, "precision": 0.85, "recall": 0.8, "f1_score": 0.82}
    result = generate_evaluation_insights(metrics)
    assert "The model has strong overall accuracy" in result["insights"]


def test_low_accuracy():
    metrics = {"accuracy": 0.3, "precision": 0.4, "recall": 0.5, "f1_score": 0.45}
    result = generate_evaluation_insights(metrics)
    assert "Overall accuracy needs improvement" in result["insights"]


def test_precision_recall_balance():
    metrics = {"accuracy": 0.7, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
    result = generate_evaluation_insights(metrics)
    assert "Precision and recall are balanced" in result["insights"]


def test_false_positive_insight():
    metrics = {"accuracy": 0.7, "precision": 0.6, "recall": 0.8, "f1_score": 0.7}
    result = generate_evaluation_insights(metrics)
    assert any("false positives" in item for item in result["insights"])


def test_false_negative_insight():
    metrics = {"accuracy": 0.7, "precision": 0.9, "recall": 0.6, "f1_score": 0.7}
    result = generate_evaluation_insights(metrics)
    assert any("misses relatively more" in item for item in result["insights"])


def test_strongest_metric():
    metrics = {"accuracy": 0.9, "precision": 0.7, "recall": 0.6, "f1_score": 0.8}
    result = generate_evaluation_insights(metrics)
    assert result["strongest_metric"] == "accuracy"


def test_weakest_metric():
    metrics = {"accuracy": 0.9, "precision": 0.7, "recall": 0.6, "f1_score": 0.8}
    result = generate_evaluation_insights(metrics)
    assert result["weakest_metric"] == "recall"


def test_missing_metric():
    with pytest.raises(ValueError):
        generate_evaluation_insights({"accuracy": 0.8})


def test_invalid_input():
    with pytest.raises(TypeError):
        generate_evaluation_insights([])


def test_invalid_metric_range():
    metrics = {"accuracy": 1.5, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
    with pytest.raises(ValueError):
        generate_evaluation_insights(metrics)