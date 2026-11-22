import pytest
from prediction_consistency import analyze_prediction_consistency


def test_perfect_consistency():
    result = analyze_prediction_consistency([[1, 1, 1]])
    assert result["average_consistency"] == 1.0
    assert result["sample_results"][0]["consistent"] is True


def test_partial_consistency():
    result = analyze_prediction_consistency([[1, 1, 2, 1]])
    assert result["average_consistency"] == 0.75


def test_multiple_samples():
    result = analyze_prediction_consistency([[1, 1], [2, 2]])
    assert result["total_samples"] == 2


def test_different_predictions():
    result = analyze_prediction_consistency([[1, 2, 3]])
    assert result["average_consistency"] == round(1 / 3, 4)


def test_most_common_prediction():
    result = analyze_prediction_consistency([[2, 1, 2, 2]])
    assert result["sample_results"][0]["most_common_prediction"] == 2


def test_empty_predictions():
    with pytest.raises(ValueError):
        analyze_prediction_consistency([])


def test_empty_sample():
    with pytest.raises(ValueError):
        analyze_prediction_consistency([[]])


def test_invalid_input():
    with pytest.raises(TypeError):
        analyze_prediction_consistency("predictions")


def test_invalid_sample_type():
    with pytest.raises(TypeError):
        analyze_prediction_consistency([1, 2])


def test_none_prediction():
    with pytest.raises(ValueError):
        analyze_prediction_consistency([[1, None]])