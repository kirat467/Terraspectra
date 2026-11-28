import numpy as np
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


def test_infinite_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": float("inf")}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_negative_infinite_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": float("-inf")}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_negative_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": -1}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_boolean_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": True}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_string_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": "90"}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_invalid_prediction_type():
    with pytest.raises(ValueError):
        generate_disease_analytics("Healthy")


def test_invalid_prediction_record():
    with pytest.raises(ValueError):
        generate_disease_analytics(["Healthy"])


def test_missing_prediction_key():
    predictions = [{"confidence": 90}]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_missing_confidence_key():
    predictions = [{"prediction": "Healthy"}]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_whitespace_disease_name():
    predictions = [
        {"prediction": "   ", "confidence": 90}
    ]

    with pytest.raises(ValueError):
        generate_disease_analytics(predictions)


def test_confidence_boundaries():
    predictions = [
        {"prediction": "Healthy", "confidence": 0},
        {"prediction": "Blight", "confidence": 100}
    ]

    result = generate_disease_analytics(predictions)

    assert result["total_predictions"] == 2


def test_numpy_confidence():
    predictions = [
        {"prediction": "Healthy", "confidence": np.float64(90)}
    ]

    result = generate_disease_analytics(predictions)

    assert result["disease_analytics"][0]["average_confidence"] == 90


def test_disease_sorting():
    predictions = [
        {"prediction": "Healthy", "confidence": 90},
        {"prediction": "Blight", "confidence": 80},
        {"prediction": "Blight", "confidence": 85},
        {"prediction": "Healthy", "confidence": 95},
        {"prediction": "Healthy", "confidence": 88}
    ]

    result = generate_disease_analytics(predictions)

    assert result["disease_analytics"][0]["disease"] == "Healthy"
    assert result["disease_analytics"][0]["count"] == 3
    assert result["disease_analytics"][1]["disease"] == "Blight"


def test_tie_preserves_insertion_order():
    predictions = [
        {"prediction": "Healthy", "confidence": 90},
        {"prediction": "Blight", "confidence": 80}
    ]

    result = generate_disease_analytics(predictions)

    assert result["disease_analytics"][0]["disease"] == "Healthy"


def test_disease_name_whitespace_normalization():
    predictions = [
        {"prediction": " Healthy ", "confidence": 90},
        {"prediction": "Healthy", "confidence": 80}
    ]

    result = generate_disease_analytics(predictions)

    assert result["total_predictions"] == 2
    assert len(result["disease_analytics"]) == 1
    assert result["disease_analytics"][0]["count"] == 2