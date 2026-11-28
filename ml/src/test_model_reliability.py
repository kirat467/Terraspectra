import pytest
from model_reliability import analyze_model_reliability


def test_reliability_analysis():
    result = analyze_model_reliability([0.9, 0.8, 0.7, 0.4])

    assert result["total_predictions"] == 4
    assert result["reliable_predictions"] == 3
    assert result["uncertain_predictions"] == 1
    assert result["reliability_percentage"] == 75.0


def test_average_confidence():
    result = analyze_model_reliability([0.8, 0.6, 1.0])
    assert result["average_confidence"] == 0.8


def test_custom_threshold():
    result = analyze_model_reliability(
        [0.5, 0.7, 0.9], confidence_threshold=0.8
    )
    assert result["reliable_predictions"] == 1
    assert result["uncertain_predictions"] == 2


def test_low_confidence_recommendation():
    result = analyze_model_reliability([0.2, 0.3, 0.4])
    assert any("Improve" in item for item in result["recommendations"])


def test_uncertain_prediction_recommendation():
    result = analyze_model_reliability([0.2, 0.3, 0.4, 0.9])
    assert any("manually" in item for item in result["recommendations"])


def test_empty_predictions():
    with pytest.raises(ValueError):
        analyze_model_reliability([])


def test_invalid_confidence():
    with pytest.raises(ValueError):
        analyze_model_reliability([0.5, 1.5])


def test_invalid_threshold():
    with pytest.raises(ValueError):
        analyze_model_reliability([0.5], confidence_threshold=2)