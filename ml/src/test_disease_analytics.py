import pytest
from disease_analytics import generate_disease_analytics


def test_disease_analytics():
    predictions = [
        {"prediction": "Healthy", "confidence": 95},
        {"prediction": "Blight", "confidence": 80},
        {"prediction": "Healthy", "confidence": 90},
        {"prediction": "Leaf Rust", "confidence": 70},
        {"prediction": "Blight", "confidence": 85}
    ]

    result = generate_disease_analytics(predictions)

    assert result["total_predictions"] == 5
    assert result["most_detected_disease"] == "Healthy"

    healthy = result["disease_analytics"][0]

    assert healthy["count"] == 2
    assert healthy["percentage"] == 40
    assert healthy["average_confidence"] == 92.5
    assert healthy["highest_confidence"] == 95
    assert healthy["lowest_confidence"] == 90


def test_empty_predictions():
    result = generate_disease_analytics([])

    assert result["total_predictions"] == 0
    assert result["most_detected_disease"] is None
    assert result["disease_analytics"] == []


def test_disease_percentages():
    predictions = [
        {"prediction": "Healthy", "confidence": 90},
        {"prediction": "Blight", "confidence": 80}
    ]

    result = generate_disease_analytics(predictions)

    assert result["disease_analytics"][0]["percentage"] == 50
    assert result["disease_analytics"][1]["percentage"] == 50


def test_invalid_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": 150}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_invalid_disease():
    predictions = [
        {"prediction": "", "confidence": 90}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_nan_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": float("nan")}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)