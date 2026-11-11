import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from ml.src.visualize_spectral import (
    plot_spectral_signature,
    plot_band_images
)


def test_spectral_signature():
    cube = np.random.rand(10, 12, 20)
    fig = plot_spectral_signature(cube, 5, 5)
    assert fig is not None


def test_band_images():
    cube = np.random.rand(10, 12, 20)
    fig = plot_band_images(cube, bands=(2, 5, 10))
    assert fig is not None


def test_invalid_pixel():
    cube = np.random.rand(10, 12, 20)

    with pytest.raises(ValueError):
        plot_spectral_signature(cube, 20, 5)


def test_invalid_band():
    cube = np.random.rand(10, 12, 20)

    with pytest.raises(ValueError):
        plot_band_images(cube, bands=(2, 5, 25))