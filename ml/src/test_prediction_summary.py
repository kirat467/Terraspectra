import pytest
from prediction_summary import generate_prediction_summary


def test_prediction_summary():
    predictions = [
        {"prediction": "Healthy", "confidence": 95},
        {"prediction": "Leaf Rust", "confidence": 85},
        {"prediction": "Blight", "confidence": 75},
        {"prediction": "Healthy", "confidence": 90}
    ]

    result = generate_prediction_summary(predictions)

    assert result["total_predictions"] == 4
    assert result["healthy_count"] == 2
    assert result["diseased_count"] == 2
    assert result["most_common_disease"] == "Healthy"
    assert result["average_confidence"] == 86.25
    assert result["healthy_percentage"] == 50
    assert result["diseased_percentage"] == 50


def test_empty_predictions():
    result = generate_prediction_summary([])

    assert result["total_predictions"] == 0
    assert result["healthy_count"] == 0
    assert result["diseased_count"] == 0
    assert result["average_confidence"] == 0


def test_all_healthy():
    predictions = [
        {"prediction": "Healthy", "confidence": 90},
        {"prediction": "Healthy", "confidence": 95}
    ]

    result = generate_prediction_summary(predictions)

    assert result["healthy_count"] == 2
    assert result["diseased_count"] == 0
    assert result["healthy_percentage"] == 100


def test_invalid_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": 150}
    ]

    with pytest.raises(ValueError):
        generate_prediction_summary(predictions)


def test_invalid_disease():
    predictions = [
        {"prediction": "", "confidence": 90}
    ]

    with pytest.raises(ValueError):
        generate_prediction_summary(predictions)