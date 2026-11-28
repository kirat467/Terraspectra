import pytest
import numpy as np

from evaluation_trends import analyze_evaluation_trends


def sample_history():
    return [
        {
            "accuracy": 0.6,
            "precision": 0.5,
            "recall": 0.6,
            "f1_score": 0.55
        },
        {
            "accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.7,
            "f1_score": 0.65
        },
        {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.8,
            "f1_score": 0.75
        }
    ]


def test_improving_trend():
    result = analyze_evaluation_trends(sample_history())
    assert result["accuracy"]["trend"] == "Improving"


def test_declining_trend():
    history = sample_history()
    history.reverse()

    result = analyze_evaluation_trends(history)

    assert result["accuracy"]["trend"] == "Declining"


def test_stable_trend():
    history = [sample_history()[0], sample_history()[0]]

    result = analyze_evaluation_trends(history)

    assert result["accuracy"]["trend"] == "Stable"


def test_latest_value():
    result = analyze_evaluation_trends(sample_history())

    assert result["accuracy"]["latest_value"] == 0.8


def test_first_value():
    result = analyze_evaluation_trends(sample_history())

    assert result["accuracy"]["first_value"] == 0.6


def test_change_value():
    result = analyze_evaluation_trends(sample_history())

    assert result["accuracy"]["change"] == 0.2


def test_single_entry():
    result = analyze_evaluation_trends([sample_history()[0]])

    assert result["accuracy"]["trend"] == "Insufficient data"


def test_empty_history():
    with pytest.raises(ValueError):
        analyze_evaluation_trends([])


def test_invalid_input():
    with pytest.raises(TypeError):
        analyze_evaluation_trends("history")


def test_missing_metric():
    with pytest.raises(ValueError):
        analyze_evaluation_trends([{"accuracy": 0.8}])


def test_invalid_metric_range():
    history = sample_history()
    history[0]["accuracy"] = 1.5

    with pytest.raises(ValueError):
        analyze_evaluation_trends(history)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_out_of_range_values(value):
    history = sample_history()
    history[0]["accuracy"] = value

    with pytest.raises(ValueError):
        analyze_evaluation_trends(history)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values(value):
    history = sample_history()
    history[0]["accuracy"] = value

    with pytest.raises(ValueError):
        analyze_evaluation_trends(history)


@pytest.mark.parametrize("value", [True, False, "0.8", None, 1 + 2j])
def test_invalid_metric_types(value):
    history = sample_history()
    history[0]["accuracy"] = value

    with pytest.raises(TypeError):
        analyze_evaluation_trends(history)


def test_numpy_numeric_values():
    history = sample_history()
    history[0]["accuracy"] = np.float64(0.65)

    result = analyze_evaluation_trends(history)

    assert result["accuracy"]["first_value"] == 0.65


def test_all_metrics_are_returned():
    result = analyze_evaluation_trends(sample_history())

    assert set(result.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    }


def test_output_structure():
    result = analyze_evaluation_trends(sample_history())

    assert set(result["accuracy"].keys()) == {
        "values",
        "first_value",
        "latest_value",
        "change",
        "trend"
    }


def test_original_history_is_not_modified():
    history = sample_history()
    original = [entry.copy() for entry in history]

    analyze_evaluation_trends(history)

    assert history == original


def test_all_metrics_are_validated():
    history = sample_history()
    history[1]["f1_score"] = float("nan")

    with pytest.raises(ValueError):
        analyze_evaluation_trends(history)


def test_two_entries_with_same_values():
    entry = sample_history()[0]

    result = analyze_evaluation_trends([entry, entry.copy()])

    assert result["accuracy"]["trend"] == "Stable"
    assert result["accuracy"]["change"] == 0.0


def test_values_are_returned_as_floats():
    history = sample_history()
    history[0]["accuracy"] = 1

    result = analyze_evaluation_trends(history)

    assert isinstance(result["accuracy"]["first_value"], float)