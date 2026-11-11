import numpy as np
import pytest
from ml.src.compare_spectra import compare_spectra


def test_compare_spectra():
    healthy = np.random.rand(10, 10, 20)
    diseased = np.random.rand(10, 10, 20)

    h, d, fig = compare_spectra(healthy, diseased)

    assert h.shape == (20,)
    assert d.shape == (20,)
    assert fig is not None


def test_different_shapes():
    healthy = np.random.rand(10, 10, 20)
    diseased = np.random.rand(10, 10, 15)

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)


def test_invalid_data():
    healthy = np.random.rand(10, 10, 20)
    diseased = np.random.rand(10, 10, 20)
    diseased[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)