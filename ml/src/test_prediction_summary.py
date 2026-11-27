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
    assert result["disease_distribution"] == {}


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
    with pytest.raises(ValueError):
        generate_prediction_summary([
            {"prediction": "Healthy", "confidence": 150}
        ])


def test_invalid_disease():
    with pytest.raises(ValueError):
        generate_prediction_summary([
            {"prediction": "", "confidence": 90}
        ])


def test_invalid_input_type():
    with pytest.raises(ValueError):
        generate_prediction_summary("invalid")


def test_invalid_prediction_record():
    with pytest.raises(ValueError):
        generate_prediction_summary(["Healthy"])


def test_missing_prediction():
    with pytest.raises(ValueError):
        generate_prediction_summary([
            {"confidence": 90}
        ])


def test_missing_confidence():
    with pytest.raises(ValueError):
        generate_prediction_summary([
            {"prediction": "Healthy"}
        ])


def test_negative_confidence():
    with pytest.raises(ValueError):
        generate_prediction_summary([
            {"prediction": "Healthy", "confidence": -5}
        ])


def test_boolean_confidence():
    with pytest.raises(ValueError):
        generate_prediction_summary([
            {"prediction": "Healthy", "confidence": True}
        ])


def test_nan_confidence():
    with pytest.raises(ValueError):
        generate_prediction_summary([
            {"prediction": "Healthy", "confidence": float("nan")}
        ])


def test_infinite_confidence():
    with pytest.raises(ValueError):
        generate_prediction_summary([
            {"prediction": "Healthy", "confidence": float("inf")}
        ])


def test_disease_distribution():
    result = generate_prediction_summary([
        {"prediction": "Healthy", "confidence": 90},
        {"prediction": "Blight", "confidence": 80},
        {"prediction": "Blight", "confidence": 70}
    ])

    assert result["disease_distribution"]["Blight"]["count"] == 2
    assert result["disease_distribution"]["Blight"]["percentage"] == 66.67


def test_confidence_range():
    result = generate_prediction_summary([
        {"prediction": "Healthy", "confidence": 60},
        {"prediction": "Blight", "confidence": 90}
    ])

    assert result["minimum_confidence"] == 60
    assert result["maximum_confidence"] == 90


def test_whitespace_in_disease_name():
    result = generate_prediction_summary([
        {"prediction": " Healthy ", "confidence": 90}
    ])

    assert result["healthy_count"] == 1


def test_confidence_boundary_values():
    result = generate_prediction_summary([
        {"prediction": "Healthy", "confidence": 0},
        {"prediction": "Blight", "confidence": 100}
    ])

    assert result["average_confidence"] == 50