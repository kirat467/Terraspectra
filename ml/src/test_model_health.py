import pytest

from model_health import assess_model_health


@pytest.fixture
def metrics():
    return {
        "accuracy": 0.90,
        "precision": 0.85,
        "recall": 0.88,
        "f1_score": 0.86
    }


def test_good_model_health(metrics):
    result = assess_model_health(metrics)

    assert result["health_status"] == "Good"
    assert result["health_score"] == 87.25


def test_low_accuracy(metrics):
    metrics["accuracy"] = 0.50

    result = assess_model_health(metrics)

    assert "Improve model accuracy." in result["recommendations"]


def test_model_drift(metrics):
    result = assess_model_health(metrics, drift_detected=True)

    assert result["health_status"] == "Needs Attention"
    assert "Investigate model drift." in result["recommendations"]


def test_low_confidence(metrics):
    result = assess_model_health(metrics, average_confidence=0.4)

    assert "Review low-confidence predictions." in result["recommendations"]


def test_missing_metric():
    with pytest.raises(ValueError):
        assess_model_health({"accuracy": 0.9})


def test_invalid_metric(metrics):
    metrics["accuracy"] = 1.5

    with pytest.raises(ValueError):
        assess_model_health(metrics)


def test_invalid_confidence(metrics):
    with pytest.raises(ValueError):
        assess_model_health(metrics, average_confidence=1.5)


def test_invalid_drift_status(metrics):
    with pytest.raises(ValueError):
        assess_model_health(metrics, drift_detected="yes")