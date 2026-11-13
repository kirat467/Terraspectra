import numpy as np
import pytest

from class_analysis import analyze_class_distribution


def test_balanced_classes():
    labels = np.array([0, 0, 1, 1, 2, 2])

    result = analyze_class_distribution(labels, 3)

    assert result["class_counts"] == {
        0: 2,
        1: 2,
        2: 2
    }

    assert np.allclose(result["class_weights"], [1, 1, 1])


def test_imbalanced_classes():
    labels = np.array([0, 0, 0, 1, 2, 2])

    result = analyze_class_distribution(labels, 3)

    weights = result["class_weights"]

    assert weights[0] < weights[1]
    assert weights[1] > weights[2]
    assert weights[0] < weights[2]


def test_empty_labels():
    with pytest.raises(ValueError):
        analyze_class_distribution([], 3)