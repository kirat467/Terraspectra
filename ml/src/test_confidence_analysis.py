import pytest
from confidence_analysis import (
    classify_confidence,
    analyze_prediction_confidence
)


def test_low_confidence():
    result = classify_confidence(25)
    assert result["confidence_level"] == "Low"


def test_medium_confidence():
    result = classify_confidence(55)
    assert result["confidence_level"] == "Medium"


def test_high_confidence():
    result = classify_confidence(90)
    assert result["confidence_level"] == "High"


def test_boundary_values():
    assert classify_confidence(40)["confidence_level"] == "Medium"
    assert classify_confidence(70)["confidence_level"] == "High"


def test_analyze_predictions():
    predictions = [
        {"prediction": "Healthy", "confidence": 85},
        {"prediction": "Blight", "confidence": 35}
    ]

    results = analyze_prediction_confidence(predictions)

    assert len(results) == 2
    assert results[0]["confidence_level"] == "High"
    assert results[1]["confidence_level"] == "Low"


def test_invalid_confidence():
    with pytest.raises(ValueError):
        classify_confidence(120)


def test_empty_predictions():
    assert analyze_prediction_confidence([]) == []