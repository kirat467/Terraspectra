import numpy as np
import pytest

from class_analysis import analyze_class_distribution


def test_balanced_classes():
    labels = np.array([0, 0, 1, 1, 2, 2])

    result = analyze_class_distribution(labels, 3)

    assert result["class_counts"] == {0: 2, 1: 2, 2: 2}
    assert np.allclose(result["class_weights"], [1, 1, 1])


def test_imbalanced_classes():
    labels = np.array([0, 0, 0, 1, 2, 2])

    result = analyze_class_distribution(labels, 3)
    weights = result["class_weights"]

    assert weights[0] < weights[1]
    assert weights[1] > weights[2]
    assert weights[0] < weights[2]


def test_empty_labels():
    with pytest.raises(ValueError, match="Labels cannot be empty"):
        analyze_class_distribution([], 3)


def test_zero_classes():
    with pytest.raises(ValueError, match="positive"):
        analyze_class_distribution([0, 1], 0)


def test_negative_class_count():
    with pytest.raises(ValueError, match="positive"):
        analyze_class_distribution([0, 1], -1)


@pytest.mark.parametrize("num_classes", [2.5, "3", None, True])
def test_invalid_class_count_type(num_classes):
    with pytest.raises(ValueError):
        analyze_class_distribution([0, 1], num_classes)


def test_negative_labels():
    with pytest.raises(ValueError, match="Invalid class labels"):
        analyze_class_distribution([-1, 0, 1], 3)


def test_out_of_range_labels():
    with pytest.raises(ValueError, match="Invalid class labels"):
        analyze_class_distribution([0, 1, 3], 3)


def test_float_labels():
    with pytest.raises(ValueError, match="integers"):
        analyze_class_distribution([0.0, 1.0], 2)


def test_boolean_labels():
    with pytest.raises(ValueError, match="integers"):
        analyze_class_distribution([True, False], 2)


def test_multidimensional_labels():
    with pytest.raises(ValueError, match="one-dimensional"):
        analyze_class_distribution([[0, 1], [1, 0]], 2)


def test_missing_class():
    result = analyze_class_distribution([0, 0, 2, 2], 3)

    assert result["class_counts"] == {0: 2, 1: 0, 2: 2}
    assert result["class_weights"][1] == 0


def test_single_class_dataset():
    result = analyze_class_distribution([0, 0, 0], 1)

    assert result["class_counts"] == {0: 3}
    assert np.allclose(result["class_weights"], [1])


def test_numpy_integer_class_count():
    result = analyze_class_distribution([0, 1], np.int64(2))

    assert result["class_counts"] == {0: 1, 1: 1}
    assert np.allclose(result["class_weights"], [1, 1])