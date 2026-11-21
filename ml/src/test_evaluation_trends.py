import pytest
from evaluation_trends import analyze_evaluation_trends


def sample_history():
    return [
        {"accuracy": 0.6, "precision": 0.5, "recall": 0.6, "f1_score": 0.55},
        {"accuracy": 0.7, "precision": 0.6, "recall": 0.7, "f1_score": 0.65},
        {"accuracy": 0.8, "precision": 0.7, "recall": 0.8, "f1_score": 0.75}
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