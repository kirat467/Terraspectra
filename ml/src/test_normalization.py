import numpy as np
import pytest

from ml.src.normalization import normalize_spectral_cube


def test_zscore_shape():
    cube = np.random.rand(10, 10, 5)
    result = normalize_spectral_cube(cube)

    assert result.shape == cube.shape


def test_zscore_mean():
    cube = np.random.rand(10, 10, 5)
    result = normalize_spectral_cube(cube)

    assert np.allclose(result.mean(axis=(0, 1)), 0, atol=1e-5)


def test_minmax_range():
    cube = np.random.rand(10, 10, 5)
    result = normalize_spectral_cube(cube, "minmax")

    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_constant_band():
    cube = np.ones((5, 5, 3))
    result = normalize_spectral_cube(cube, "minmax")

    assert np.isfinite(result).all()


def test_invalid_method():
    with pytest.raises(ValueError):
        normalize_spectral_cube(np.ones((5, 5, 3)), "invalid")


def test_invalid_cube():
    with pytest.raises(ValueError):
        normalize_spectral_cube(np.ones((5, 5)))


def test_snv_normalization():
    cube = np.random.rand(10, 10, 550)
    result = normalize_spectral_cube(cube, "snv")

    assert result.shape == cube.shape
    assert np.allclose(result.mean(axis=-1), 0, atol=1e-5)
    assert np.allclose(result.std(axis=-1), 1, atol=1e-5)


def test_snv_constant_spectrum():
    cube = np.ones((5, 5, 10))
    result = normalize_spectral_cube(cube, "snv")

    assert np.isfinite(result).all()
    assert np.all(result == 0)


def test_invalid_values():
    cube = np.ones((5, 5, 3))
    cube[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        normalize_spectral_cube(cube)