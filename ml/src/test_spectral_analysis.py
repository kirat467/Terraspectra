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
    assert result["healthy_mean"][2] == 1
    assert result["diseased_mean"][2] == 3


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


def test_infinite_values():
    healthy = np.ones((4, 4, 5))
    diseased = np.ones((4, 4, 5))

    diseased[0, 0, 0] = np.inf

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased)


def test_empty_cube():
    healthy = np.empty((0, 4, 5))
    diseased = np.empty((0, 4, 5))

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased)


def test_non_numeric_cube():
    healthy = np.full((2, 2, 5), "abc")
    diseased = np.ones((2, 2, 5))

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased)


def test_complex_values():
    healthy = np.ones((2, 2, 5), dtype=complex)
    diseased = np.ones((2, 2, 5))

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased)


@pytest.mark.parametrize("top_k", [0, -1, 6])
def test_out_of_range_top_k(top_k):
    healthy = np.ones((2, 2, 5))
    diseased = np.ones((2, 2, 5))

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased, top_k=top_k)


@pytest.mark.parametrize("top_k", [2.5, "2", True, None])
def test_non_integer_top_k(top_k):
    healthy = np.ones((2, 2, 5))
    diseased = np.ones((2, 2, 5))

    with pytest.raises(ValueError):
        analyze_spectral_difference(healthy, diseased, top_k=top_k)


def test_top_k_equal_to_number_of_bands():
    healthy = np.zeros((2, 2, 5))
    diseased = np.ones((2, 2, 5))

    result = analyze_spectral_difference(
        healthy, diseased, top_k=5
    )

    assert len(result["top_bands"]) == 5