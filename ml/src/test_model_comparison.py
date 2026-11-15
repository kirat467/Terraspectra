import pytest

from model_comparison import compare_models


def test_model_comparison():

    cnn = {
        "accuracy": 0.85,
        "precision": 0.84,
        "recall": 0.83,
        "f1_score": 0.82
    }

    vit = {
        "accuracy": 0.90,
        "precision": 0.89,
        "recall": 0.88,
        "f1_score": 0.87
    }

    result = compare_models(cnn, vit)

    assert result["best_model"] == "ViT"
    assert result["selection_metric"] == "f1_score"


def test_cnn_wins():

    cnn = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.92
    }

    vit = {
        "accuracy": 0.80,
        "precision": 0.79,
        "recall": 0.78,
        "f1_score": 0.77
    }

    result = compare_models(cnn, vit)

    assert result["best_model"] == "3D-CNN"


def test_equal_models():

    metrics = {
        "accuracy": 0.90,
        "precision": 0.90,
        "recall": 0.90,
        "f1_score": 0.90
    }

    result = compare_models(metrics, metrics)

    assert result["best_model"] == "Tie"


def test_invalid_metric():

    cnn = {
        "accuracy": 1.5,
        "precision": 0.8,
        "recall": 0.8,
        "f1_score": 0.8
    }

    vit = {
        "accuracy": 0.9,
        "precision": 0.9,
        "recall": 0.9,
        "f1_score": 0.9
    }

    with pytest.raises(ValueError):
        compare_models(cnn, vit)


def test_metric_difference():

    cnn = {
        "accuracy": 0.80,
        "precision": 0.80,
        "recall": 0.80,
        "f1_score": 0.80
    }

    vit = {
        "accuracy": 0.90,
        "precision": 0.90,
        "recall": 0.90,
        "f1_score": 0.90
    }

    result = compare_models(cnn, vit)

    assert result["model_comparison"]["accuracy"]["difference"] == -0.1
    assert result["model_comparison"]["accuracy"]["winner"] == "ViT"