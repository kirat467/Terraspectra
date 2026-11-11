import numpy as np
import pytest
import torch

from dataset import HyperspectralDataset


def test_dataset_without_normalization():
    images = [np.random.rand(10, 10, 5)]
    dataset = HyperspectralDataset(images, [1])

    image, label = dataset[0]

    assert image.shape == (1, 5, 10, 10)
    assert label.item() == 1


def test_zscore_dataset():
    images = [np.random.rand(10, 10, 5)]
    dataset = HyperspectralDataset(
        images, [0], normalization="zscore"
    )

    image, label = dataset[0]

    assert image.dtype == torch.float32
    assert abs(image.mean().item()) < 1e-5


def test_minmax_dataset():
    images = [np.random.rand(10, 10, 5)]
    dataset = HyperspectralDataset(
        images, [1], normalization="minmax"
    )

    image, label = dataset[0]

    assert image.min() >= 0
    assert image.max() <= 1


def test_invalid_normalization():
    with pytest.raises(ValueError):
        HyperspectralDataset(
            [np.ones((5, 5, 3))], [0],
            normalization="invalid"
        )


def test_dataset_length():
    images = [
        np.random.rand(5, 5, 3),
        np.random.rand(5, 5, 3)
    ]

    dataset = HyperspectralDataset(
        images, [0, 1], normalization="zscore"
    )

    assert len(dataset) == 2