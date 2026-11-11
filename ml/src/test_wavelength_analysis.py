import numpy as np
import pytest

from wavelength_analysis import (
    band_to_wavelength,
    rank_wavelength_differences
)


def test_band_to_wavelength():
    result = band_to_wavelength(
        np.array([0, 549])
    )

    assert np.isclose(result[0], 400)
    assert np.isclose(result[1], 950)


def test_rank_wavelength_differences():
    difference = np.array([0.1, 0.5, 0.2, 0.9, 0.3])

    result = rank_wavelength_differences(
        difference, top_k=3
    )

    assert len(result["top_bands"]) == 3
    assert result["top_bands"][0] == 3
    assert result["differences"][0] == 0.9


def test_invalid_band():
    with pytest.raises(ValueError):
        band_to_wavelength(np.array([600]))


def test_invalid_top_k():
    difference = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        rank_wavelength_differences(difference, top_k=5)


def test_invalid_difference():
    difference = np.array([0.1, np.nan, 0.3])

    with pytest.raises(ValueError):
        rank_wavelength_differences(difference)