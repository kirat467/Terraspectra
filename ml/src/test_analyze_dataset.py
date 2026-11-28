import numpy as np
import pytest

from analyze_dataset import analyze_dataset


def test_valid_dataset():
    images = np.random.rand(100, 16, 16, 200)
    labels = (
        ["Healthy"] * 40
        + ["Leaf Rust"] * 35
        + ["Blight"] * 25
    )

    report = analyze_dataset(images, labels)

    assert len(report) == 3
    assert report["Healthy"] == 40
    assert report["Leaf Rust"] == 35
    assert report["Blight"] == 25
    assert sum(report.values()) == 100


def test_missing_labels():
    images = np.random.rand(4, 16, 16, 10)
    labels = ["Healthy", None, np.nan, "Blight"]

    report = analyze_dataset(images, labels)

    assert report == {"Healthy": 1, "Blight": 1}


def test_empty_dataset():
    with pytest.raises(ValueError, match="Dataset is empty"):
        analyze_dataset([], [])


def test_mismatched_lengths():
    images = np.random.rand(3, 16, 16, 10)
    labels = ["Healthy", "Blight"]

    with pytest.raises(ValueError, match="equal length"):
        analyze_dataset(images, labels)


def test_none_images():
    with pytest.raises(ValueError, match="Images cannot be None"):
        analyze_dataset(None, ["Healthy"])


def test_none_labels():
    images = np.random.rand(1, 16, 16, 10)

    with pytest.raises(ValueError, match="Labels cannot be None"):
        analyze_dataset(images, None)


def test_non_numeric_images():
    images = np.array([[["invalid"]]])
    
    with pytest.raises(ValueError, match="numeric"):
        analyze_dataset(images, ["Healthy"])


def test_non_finite_images():
    images = np.random.rand(2, 16, 16, 10)
    images[0, 0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        analyze_dataset(images, ["Healthy", "Blight"])


def test_images_with_insufficient_dimensions():
    images = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        analyze_dataset(images, ["Healthy", "Blight", "Rust"])


def test_unhashable_labels():
    images = np.random.rand(2, 16, 16, 10)
    labels = [["Healthy"], ["Blight"]]

    with pytest.raises(ValueError, match="hashable"):
        analyze_dataset(images, labels)


def test_all_labels_missing():
    images = np.random.rand(2, 16, 16, 10)
    labels = [None, np.nan]

    report = analyze_dataset(images, labels)

    assert len(report) == 0
    assert sum(report.values()) == 0