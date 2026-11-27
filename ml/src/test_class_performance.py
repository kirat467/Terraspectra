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
    result = analyze_class_performance(
        [0, 0, 1, 1],
        [0, 1, 1, 1]
    )

    assert result["class_performance"][0]["f1_score"] == 0.6667


def test_support_count():
    result = analyze_class_performance(
        [0, 0, 1],
        [0, 1, 1]
    )

    assert result["class_performance"][0]["support"] == 2
    assert result["class_performance"][1]["support"] == 1


def test_best_and_weakest_class():
    result = analyze_class_performance(
        [0, 0, 1, 1],
        [0, 0, 1, 0]
    )

    assert result["best_performing_class"] == 0
    assert result["weakest_performing_class"] == 1


def test_perfect_predictions():
    result = analyze_class_performance(
        [0, 1, 2],
        [0, 1, 2]
    )

    for metrics in result["class_performance"].values():
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0


def test_empty_input():
    with pytest.raises(ValueError, match="empty"):
        analyze_class_performance([], [])


def test_different_lengths():
    with pytest.raises(ValueError, match="equal length"):
        analyze_class_performance([0, 1], [0])


def test_invalid_input():
    with pytest.raises(ValueError, match="lists or tuples"):
        analyze_class_performance("012", [0, 1, 2])


def test_tuple_inputs():
    result = analyze_class_performance(
        (0, 0, 1, 1),
        (0, 1, 1, 1)
    )

    assert len(result["class_performance"]) == 2


def test_unhashable_labels():
    with pytest.raises(ValueError, match="hashable"):
        analyze_class_performance(
            [[0], [1]],
            [[0], [1]]
        )


@pytest.mark.parametrize("missing_label", [None, float("nan")])
def test_missing_labels(missing_label):
    with pytest.raises(ValueError, match="missing"):
        analyze_class_performance(
            [0, missing_label],
            [0, 1]
        )


def test_class_present_only_in_predictions():
    result = analyze_class_performance(
        [0, 0, 0],
        [0, 1, 1]
    )

    assert result["class_performance"][1]["precision"] == 0.0
    assert result["class_performance"][1]["recall"] == 0.0
    assert result["class_performance"][1]["support"] == 0


def test_class_with_no_correct_predictions():
    result = analyze_class_performance(
        [0, 0, 1, 1],
        [1, 1, 0, 0]
    )

    for metrics in result["class_performance"].values():
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1_score"] == 0.0


def test_string_class_labels():
    result = analyze_class_performance(
        ["Healthy", "Blight", "Healthy"],
        ["Healthy", "Healthy", "Blight"]
    )

    assert "Healthy" in result["class_performance"]
    assert "Blight" in result["class_performance"]


def test_single_class_perfect_predictions():
    result = analyze_class_performance([0, 0, 0], [0, 0, 0])

    assert result["best_performing_class"] == 0
    assert result["weakest_performing_class"] == 0
    assert result["class_performance"][0]["f1_score"] == 1.0