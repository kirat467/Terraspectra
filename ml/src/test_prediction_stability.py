import pytest
from prediction_stability import analyze_prediction_stability


def test_perfect_stability():
    result = analyze_prediction_stability([[1, 1, 1]])
    assert result["average_stability"] == 1.0
    assert result["sample_results"][0]["stable"] is True


def test_partial_stability():
    result = analyze_prediction_stability([[1, 1, 2, 1]])
    assert result["average_stability"] == 0.75


def test_different_predictions():
    result = analyze_prediction_stability([[1, 2, 3]])
    assert result["average_stability"] == round(1 / 3, 4)


def test_multiple_samples():
    result = analyze_prediction_stability([[1, 1], [2, 2]])
    assert result["total_samples"] == 2


def test_unique_predictions():
    result = analyze_prediction_stability([[1, 2, 2, 3]])
    assert result["sample_results"][0]["unique_predictions"] == 3


def test_stable_flag():
    result = analyze_prediction_stability([[1, 1, 1], [1, 2, 1]])
    assert result["sample_results"][0]["stable"] is True
    assert result["sample_results"][1]["stable"] is False


def test_empty_predictions():
    with pytest.raises(ValueError):
        analyze_prediction_stability([])


def test_empty_sample():
    with pytest.raises(ValueError):
        analyze_prediction_stability([[]])


def test_invalid_input():
    with pytest.raises(TypeError):
        analyze_prediction_stability("predictions")


def test_none_prediction():
    with pytest.raises(ValueError):
        analyze_prediction_stability([[1, None]])