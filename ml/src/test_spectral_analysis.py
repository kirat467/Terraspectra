import numpy as np
import pytest

from spectral_analysis import analyze_spectral_difference


def test_spectral_difference():
    healthy = np.ones((4, 4, 5))
    diseased = np.ones((4, 4, 5))

    diseased[:, :, 2] = 3

    result = analyze_spectral_difference(
        healthy, diseased, top_k=3
    )

    assert result["difference"].shape == (5,)
    assert result["top_bands"][0] == 2


def test_top_k():
    healthy = np.zeros((2, 2, 5))
    diseased = np.ones((2, 2, 5))

    result = analyze_spectral_difference(
        healthy, diseased, top_k=3
    )

    assert len(result["top_bands"]) == 3


def test_invalid_shape():
    healthy = np.ones((4, 4, 5))
    diseased = np.ones((4, 4, 6))

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased)


def test_invalid_top_k():
    healthy = np.ones((4, 4, 5))
    diseased = np.ones((4, 4, 5))

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased, top_k=10)


def test_invalid_values():
    healthy = np.ones((4, 4, 5))
    diseased = np.ones((4, 4, 5))

    diseased[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased)