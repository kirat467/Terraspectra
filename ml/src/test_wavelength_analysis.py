import numpy as np
import pytest

from wavelength_analysis import (
    band_to_wavelength,
    rank_wavelength_differences
)


def test_band_to_wavelength():
    result = band_to_wavelength(np.array([0, 549]))

    assert np.isclose(result[0], 400)
    assert np.isclose(result[1], 950)


def test_rank_wavelength_differences():
    difference = np.array([0.1, 0.5, 0.2, 0.9, 0.3])

    result = rank_wavelength_differences(difference, top_k=3)

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


def test_empty_band_indices():
    with pytest.raises(ValueError):
        band_to_wavelength(np.array([]))


@pytest.mark.parametrize("total_bands", [0, 1, -5])
def test_invalid_total_bands(total_bands):
    with pytest.raises(ValueError):
        band_to_wavelength(np.array([0]), total_bands=total_bands)


@pytest.mark.parametrize("total_bands", [5.5, True, "550"])
def test_non_integer_total_bands(total_bands):
    with pytest.raises(ValueError):
        band_to_wavelength(np.array([0]), total_bands=total_bands)


def test_invalid_wavelength_range():
    with pytest.raises(ValueError):
        band_to_wavelength(np.array([0]), start_nm=950, end_nm=400)


@pytest.mark.parametrize(
    "start_nm,end_nm",
    [(np.nan, 950), (400, np.inf), (400, np.nan)]
)
def test_non_finite_wavelength_range(start_nm, end_nm):
    with pytest.raises(ValueError):
        band_to_wavelength(
            np.array([0]),
            start_nm=start_nm,
            end_nm=end_nm
        )


def test_non_numeric_difference():
    difference = np.array(["a", "b", "c"])

    with pytest.raises(ValueError):
        rank_wavelength_differences(difference)


def test_complex_difference():
    difference = np.array([1 + 2j, 2 + 3j])

    with pytest.raises(ValueError):
        rank_wavelength_differences(difference)


@pytest.mark.parametrize("top_k", [0, -1, 6])
def test_out_of_range_top_k(top_k):
    difference = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        rank_wavelength_differences(difference, top_k=top_k)


@pytest.mark.parametrize("top_k", [2.5, "2", True, None])
def test_non_integer_top_k(top_k):
    difference = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        rank_wavelength_differences(difference, top_k=top_k)


def test_empty_difference():
    with pytest.raises(ValueError):
        rank_wavelength_differences(np.array([]))


def test_infinite_difference():
    difference = np.array([0.1, np.inf, 0.3])

    with pytest.raises(ValueError):
        rank_wavelength_differences(difference)


def test_top_k_equal_to_number_of_bands():
    difference = np.array([0.1, 0.5, 0.2])

    result = rank_wavelength_differences(difference, top_k=3)

    assert len(result["top_bands"]) == 3