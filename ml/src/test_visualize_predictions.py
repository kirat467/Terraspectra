import numpy as np
import pytest

from visualize_predictions import visualize_predictions


def test_prediction_visualizations(tmp_path):
    results = {
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

    paths = visualize_predictions(results, tmp_path)

    assert len(paths) == 3

    for path in paths.values():
        assert (tmp_path / path.split("\\")[-1]).exists() or \
               __import__("pathlib").Path(path).exists()


def test_invalid_confidence(tmp_path):
    results = {
        "class_labels": [0],
        "confusion_matrix": [[2]],
        "prediction_counts": {0: 2},
        "mean_confidence": 1.5
    }

    with pytest.raises(ValueError):
        visualize_predictions(results, tmp_path)