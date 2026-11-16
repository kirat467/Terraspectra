import pytest

from model_monitoring import generate_model_monitoring


def test_complete_monitoring():
    result = generate_model_monitoring(
        prediction_summary={"total_predictions": 100},
        confidence_analysis={"average_confidence": 0.85},
        drift_report={"status": "No Significant Drift"},
        performance_report={"accuracy": 0.90}
    )

    assert result["monitoring_status"] == "Complete"
    assert result["available_sections"] == 4
    assert result["drift_status"] == "No Significant Drift"


def test_partial_monitoring():
    result = generate_model_monitoring(
        prediction_summary={"total_predictions": 50}
    )

    assert result["monitoring_status"] == "Partial"
    assert result["available_sections"] == 1
    assert len(result["recommendations"]) > 0


def test_missing_data():
    result = generate_model_monitoring()

    assert result["monitoring_status"] == "Partial"
    assert result["available_sections"] == 0
    assert result["drift_status"] == "Unknown"


def test_drift_detected():
    result = generate_model_monitoring(
        drift_report={"status": "Drift Detected"}
    )

    assert result["drift_status"] == "Drift Detected"
    assert any(
        "Investigate" in item
        for item in result["recommendations"]
    )


def test_invalid_prediction_summary():
    with pytest.raises(ValueError):
        generate_model_monitoring(prediction_summary=[])


def test_invalid_confidence_analysis():
    with pytest.raises(ValueError):
        generate_model_monitoring(confidence_analysis="invalid")


def test_invalid_drift_report():
    with pytest.raises(ValueError):
        generate_model_monitoring(drift_report=[])


def test_invalid_performance_report():
    with pytest.raises(ValueError):
        generate_model_monitoring(performance_report=[])