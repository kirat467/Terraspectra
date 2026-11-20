import pytest
from class_performance import analyze_class_performance


def test_class_metrics():
    result = analyze_class_performance(
        [0, 0, 1, 1],
        [0, 1, 1, 1]
    )

    assert result["class_performance"][0]["precision"] == 1.0
    assert result["class_performance"][0]["recall"] == 0.5
    assert result["class_performance"][1]["precision"] == 0.6667
    assert result["class_performance"][1]["recall"] == 1.0


def test_f1_score():
    result = analyze_class_performance([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["class_performance"][0]["f1_score"] == 0.6667


def test_support_count():
    result = analyze_class_performance([0, 0, 1], [0, 1, 1])
    assert result["class_performance"][0]["support"] == 2
    assert result["class_performance"][1]["support"] == 1


def test_best_and_weakest_class():
    result = analyze_class_performance([0, 0, 1, 1], [0, 0, 1, 0])
    assert result["best_performing_class"] == 0
    assert result["weakest_performing_class"] == 1


def test_perfect_predictions():
    result = analyze_class_performance([0, 1, 2], [0, 1, 2])

    for metrics in result["class_performance"].values():
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0


def test_empty_input():
    with pytest.raises(ValueError):
        analyze_class_performance([], [])


def test_different_lengths():
    with pytest.raises(ValueError):
        analyze_class_performance([0, 1], [0])


def test_invalid_input():
    with pytest.raises(ValueError):
        analyze_class_performance("012", [0, 1, 2])