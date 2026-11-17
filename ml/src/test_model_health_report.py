import pytest
from model_health_report import generate_model_health_report


def sample_health_data():
    return {
        "health_status": "Good",
        "health_score": 87.25,
        "performance_metrics": {
            "accuracy": 0.90,
            "precision": 0.85,
            "recall": 0.88,
            "f1_score": 0.86
        },
        "average_confidence": 0.91,
        "drift_detected": False,
        "recommendations": []
    }


def test_generate_health_report():
    report = generate_model_health_report(sample_health_data())
    assert "TERRASPECTRA" in report
    assert "Health Status: Good" in report
    assert "87.25%" in report


def test_report_contains_all_metrics():
    report = generate_model_health_report(sample_health_data())
    assert "Accuracy: 0.9" in report
    assert "Precision: 0.85" in report
    assert "Recall: 0.88" in report
    assert "F1 Score: 0.86" in report


def test_report_shows_drift():
    data = sample_health_data()
    data["drift_detected"] = True

    report = generate_model_health_report(data)
    assert "Detected" in report


def test_report_shows_recommendations():
    data = sample_health_data()
    data["recommendations"] = ["Improve model accuracy"]

    report = generate_model_health_report(data)
    assert "Improve model accuracy" in report


def test_report_without_recommendations():
    report = generate_model_health_report(sample_health_data())
    assert "No recommendations" in report


def test_missing_field():
    data = sample_health_data()
    del data["health_status"]

    with pytest.raises(ValueError):
        generate_model_health_report(data)