import pytest
from evaluation_comparison import compare_evaluations


def test_model_a_wins():
    a = {"accuracy": 0.9, "precision": 0.9, "recall": 0.8, "f1_score": 0.85}
    b = {"accuracy": 0.8, "precision": 0.8, "recall": 0.7, "f1_score": 0.75}
    assert compare_evaluations(a, b)["winner"] == "Model A"


def test_model_b_wins():
    a = {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1_score": 0.75}
    b = {"accuracy": 0.8, "precision": 0.9, "recall": 0.9, "f1_score": 0.85}
    assert compare_evaluations(a, b)["winner"] == "Model B"


def test_accuracy_breaks_f1_tie():
    a = {"accuracy": 0.9, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
    b = {"accuracy": 0.7, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
    assert compare_evaluations(a, b)["winner"] == "Model A"


def test_complete_tie():
    a = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
    b = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
    assert compare_evaluations(a, b)["winner"] == "Tie"


def test_missing_metric():
    with pytest.raises(ValueError):
        compare_evaluations({"accuracy": 0.8}, {"accuracy": 0.9})


def test_invalid_input():
    with pytest.raises(TypeError):
        compare_evaluations([], {})


def test_invalid_metric_range():
    a = {"accuracy": 1.5, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
    b = {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8}
    with pytest.raises(ValueError):
        compare_evaluations(a, b)