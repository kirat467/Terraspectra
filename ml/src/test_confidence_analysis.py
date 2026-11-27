import numpy as np
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


@pytest.mark.parametrize(
    "confidence",
    [-1, 101, float("inf"), float("-inf"), float("nan")]
)
def test_invalid_confidence_values(confidence):
    with pytest.raises(ValueError):
        classify_confidence(confidence)


@pytest.mark.parametrize(
    "confidence",
    ["85", None, True, False, [], {}]
)
def test_non_numeric_confidence(confidence):
    with pytest.raises(ValueError):
        classify_confidence(confidence)


def test_numpy_confidence():
    result = classify_confidence(np.float64(85.567))

    assert result["confidence"] == 85.57
    assert result["confidence_level"] == "High"


def test_missing_confidence():
    predictions = [{"prediction": "Healthy"}]

    with pytest.raises(ValueError):
        analyze_prediction_confidence(predictions)


@pytest.mark.parametrize(
    "predictions",
    ["invalid", None, {}, 123]
)
def test_invalid_predictions_input(predictions):
    with pytest.raises(ValueError):
        analyze_prediction_confidence(predictions)


def test_invalid_prediction_record():
    with pytest.raises(ValueError):
        analyze_prediction_confidence(["Healthy"])


def test_original_prediction_is_not_modified():
    prediction = {"prediction": "Healthy", "confidence": 85}
    original = prediction.copy()

    analyze_prediction_confidence([prediction])

    assert prediction == original


def test_confidence_rounding():
    result = classify_confidence(85.6789)

    assert result["confidence"] == 85.68


def test_zero_confidence():
    assert classify_confidence(0)["confidence_level"] == "Low"


def test_hundred_confidence():
    assert classify_confidence(100)["confidence_level"] == "High"