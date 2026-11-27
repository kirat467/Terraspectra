import numpy as np
import pytest

from pathlib import Path

from visualize_predictions import visualize_predictions


def get_valid_results():
    return {
        "class_labels": [0, 1, 2],
        "confusion_matrix": [
            [2, 1, 0],
            [0, 2, 1],
            [0, 1, 2]
        ],
        "prediction_counts": {
            0: 2,
            1: 4,
            2: 3
        },
        "mean_confidence": 0.75
    }


def test_prediction_visualizations(tmp_path):

    paths = visualize_predictions(get_valid_results(), tmp_path)

    assert len(paths) == 3

    for path in paths.values():
        assert Path(path).exists()


def test_invalid_confidence(tmp_path):

    results = get_valid_results()
    results["mean_confidence"] = 1.5

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf"), True, "0.5", None]
)
def test_invalid_confidence_values(tmp_path, confidence):

    results = get_valid_results()
    results["mean_confidence"] = confidence

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


def test_missing_required_key(tmp_path):

    results = get_valid_results()
    del results["class_labels"]

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


def test_invalid_results_type(tmp_path):

    with pytest.raises(ValueError):
        visualize_predictions([], tmp_path)


def test_invalid_confusion_matrix_shape(tmp_path):

    results = get_valid_results()
    results["confusion_matrix"] = [[2, 1]]

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


@pytest.mark.parametrize(
    "matrix",
    [
        [[2, -1, 0], [0, 2, 1], [0, 1, 2]],
        [[2, float("nan"), 0], [0, 2, 1], [0, 1, 2]],
        [[2.5, 1, 0], [0, 2, 1], [0, 1, 2]],
        [["invalid", 1, 0], [0, 2, 1], [0, 1, 2]]
    ]
)
def test_invalid_confusion_matrix_values(tmp_path, matrix):

    results = get_valid_results()
    results["confusion_matrix"] = matrix

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


def test_invalid_class_labels(tmp_path):

    results = get_valid_results()
    results["class_labels"] = [0, 0, 2]

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


def test_invalid_prediction_counts_type(tmp_path):

    results = get_valid_results()
    results["prediction_counts"] = []

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


def test_prediction_counts_mismatch(tmp_path):

    results = get_valid_results()
    results["prediction_counts"][0] = 100

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


def test_negative_prediction_counts(tmp_path):

    results = get_valid_results()
    results["prediction_counts"][0] = -1

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)


def test_unknown_class_labels(tmp_path):

    results = {
        "class_labels": [5],
        "confusion_matrix": [[2]],
        "prediction_counts": {5: 2},
        "mean_confidence": 0.8
    }

    paths = visualize_predictions(results, tmp_path)

    assert len(paths) == 3
    assert Path(paths["confusion_matrix"]).exists()


def test_output_directory_creation(tmp_path):

    output_dir = tmp_path / "new_folder" / "results"

    visualize_predictions(get_valid_results(), output_dir)

    assert output_dir.exists()


def test_zero_confidence(tmp_path):

    results = get_valid_results()
    results["mean_confidence"] = 0.0

    paths = visualize_predictions(results, tmp_path)

    assert Path(paths["prediction_confidence"]).exists()


def test_one_confidence(tmp_path):

    results = get_valid_results()
    results["mean_confidence"] = 1.0

    paths = visualize_predictions(results, tmp_path)

    assert Path(paths["prediction_confidence"]).exists()