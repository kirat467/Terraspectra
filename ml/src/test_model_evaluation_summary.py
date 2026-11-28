import pytest
import numpy as np

from model_evaluation_summary import create_evaluation_summary


def sample_metrics():
    return {
        "accuracy": 0.9,
        "precision": 0.85,
        "recall": 0.8,
        "f1_score": 0.82,
        "test_samples": 100
    }


def test_valid_summary():
    result = create_evaluation_summary(sample_metrics())

    assert result["accuracy"] == 0.9
    assert result["test_samples"] == 100


def test_rounding():
    metrics = sample_metrics()
    metrics["accuracy"] = 0.87654

    result = create_evaluation_summary(metrics)

    assert result["accuracy"] == 0.8765


def test_missing_metric():
    with pytest.raises(ValueError):
        create_evaluation_summary({"accuracy": 0.9})


def test_invalid_input():
    with pytest.raises(TypeError):
        create_evaluation_summary([])


@pytest.mark.parametrize(
    "metric",
    ["accuracy", "precision", "recall", "f1_score"]
)
@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_invalid_metric_range(metric, value):
    metrics = sample_metrics()
    metrics[metric] = value

    with pytest.raises(ValueError):
        create_evaluation_summary(metrics)


def test_negative_samples():
    metrics = sample_metrics()
    metrics["test_samples"] = -1

    with pytest.raises(ValueError):
        create_evaluation_summary(metrics)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")]
)
def test_non_finite_metrics(value):
    metrics = sample_metrics()
    metrics["accuracy"] = value

    with pytest.raises(ValueError):
        create_evaluation_summary(metrics)


@pytest.mark.parametrize(
    "value",
    [True, False, "0.9", None, 1 + 2j]
)
def test_invalid_metric_types(value):
    metrics = sample_metrics()
    metrics["accuracy"] = value

    with pytest.raises(TypeError):
        create_evaluation_summary(metrics)


@pytest.mark.parametrize(
    "value",
    [True, False, 10.0, 10.5, "100", None, 1 + 2j]
)
def test_invalid_test_samples_types(value):
    metrics = sample_metrics()
    metrics["test_samples"] = value

    with pytest.raises(TypeError):
        create_evaluation_summary(metrics)


def test_zero_test_samples():
    metrics = sample_metrics()
    metrics["test_samples"] = 0

    result = create_evaluation_summary(metrics)

    assert result["test_samples"] == 0


def test_numpy_numeric_values():
    metrics = sample_metrics()
    metrics["accuracy"] = np.float64(0.95)
    metrics["test_samples"] = np.int64(120)

    result = create_evaluation_summary(metrics)

    assert result["accuracy"] == 0.95
    assert result["test_samples"] == 120


def test_all_metrics_are_required():
    metrics = sample_metrics()

    for key in list(metrics.keys()):
        incomplete = metrics.copy()
        del incomplete[key]

        with pytest.raises(ValueError):
            create_evaluation_summary(incomplete)


def test_output_structure():
    result = create_evaluation_summary(sample_metrics())

    assert set(result.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "test_samples"
    }


def test_input_is_not_modified():
    metrics = sample_metrics()
    original = metrics.copy()

    create_evaluation_summary(metrics)

    assert metrics == original


def test_boundary_values():
    metrics = {
        "accuracy": 0,
        "precision": 1,
        "recall": 0,
        "f1_score": 1,
        "test_samples": 0
    }

    result = create_evaluation_summary(metrics)

    assert result["accuracy"] == 0.0
    assert result["precision"] == 1.0


def test_integer_metrics_are_converted_to_float():
    metrics = sample_metrics()
    metrics["accuracy"] = 1

    result = create_evaluation_summary(metrics)

    assert result["accuracy"] == 1.0
    assert isinstance(result["accuracy"], float)