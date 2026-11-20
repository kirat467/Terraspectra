import pytest
from prediction_error_analysis import analyze_prediction_errors


def test_basic_error_analysis():
    result = analyze_prediction_errors(
        [0, 1, 2, 1, 0],
        [0, 1, 1, 1, 0]
    )

    assert result["total_predictions"] == 5
    assert result["correct_predictions"] == 4
    assert result["incorrect_predictions"] == 1
    assert result["accuracy"] == 80.0
    assert result["error_rate"] == 20.0


def test_misclassified_samples():
    result = analyze_prediction_errors([0, 1, 2], [0, 2, 1])

    assert len(result["misclassified_samples"]) == 2
    assert result["misclassified_samples"][0] == {
        "actual": 1,
        "predicted": 2
    }


def test_class_analysis():
    result = analyze_prediction_errors([0, 0, 1, 1], [0, 1, 1, 1])

    assert result["class_analysis"][0]["error_rate"] == 50.0
    assert result["class_analysis"][1]["error_rate"] == 0.0


def test_perfect_predictions():
    result = analyze_prediction_errors([0, 1, 2], [0, 1, 2])

    assert result["accuracy"] == 100.0
    assert result["error_rate"] == 0.0
    assert result["misclassified_samples"] == []


def test_all_predictions_wrong():
    result = analyze_prediction_errors([0, 1], [1, 0])

    assert result["correct_predictions"] == 0
    assert result["incorrect_predictions"] == 2
    assert result["accuracy"] == 0.0


def test_empty_input():
    with pytest.raises(ValueError):
        analyze_prediction_errors([], [])


def test_different_lengths():
    with pytest.raises(ValueError):
        analyze_prediction_errors([0, 1], [0])


def test_invalid_input_type():
    with pytest.raises(ValueError):
        analyze_prediction_errors("012", [0, 1, 2])