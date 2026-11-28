import numpy as np
import pytest

from data_quality import analyze_data_quality


def test_data_quality():
    cube = np.random.rand(10, 10, 5).astype(np.float32)

    result = analyze_data_quality(cube)

    assert result["shape"] == (10, 10, 5)
    assert result["minimum"] >= 0
    assert result["maximum"] <= 1
    assert len(result["band_means"]) == 5
    assert len(result["band_stds"]) == 5


def test_invalid_dimensions():
    cube = np.random.rand(10, 10)

    with pytest.raises(ValueError):
        analyze_data_quality(cube)


def test_invalid_values():
    cube = np.ones((10, 10, 5))
    cube[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        analyze_data_quality(cube)


def test_empty_cube():
    cube = np.empty((0, 10, 5))

    with pytest.raises(ValueError):
        analyze_data_quality(cube)


def test_infinite_values():
    cube = np.ones((10, 10, 5))
    cube[0, 0, 0] = np.inf

    with pytest.raises(ValueError):
        analyze_data_quality(cube)


def test_negative_infinite_values():
    cube = np.ones((10, 10, 5))
    cube[0, 0, 0] = -np.inf

    with pytest.raises(ValueError):
        analyze_data_quality(cube)


def test_non_numeric_values():
    cube = np.full((10, 10, 5), "invalid")

    with pytest.raises(ValueError):
        analyze_data_quality(cube)


def test_complex_values():
    cube = np.ones((10, 10, 5), dtype=complex)

    with pytest.raises(ValueError):
        analyze_data_quality(cube)


def test_constant_bands():
    cube = np.ones((10, 10, 5))

    result = analyze_data_quality(cube)

    assert result["constant_bands"] == [0, 1, 2, 3, 4]
    assert np.all(result["band_stds"] == 0)


def test_statistical_values():
    cube = np.ones((2, 2, 2)) * 5

    result = analyze_data_quality(cube)

    assert result["minimum"] == 5
    assert result["maximum"] == 5
    assert result["mean"] == 5
    assert result["standard_deviation"] == 0


def test_band_statistics():
    cube = np.zeros((2, 2, 2))
    cube[:, :, 0] = 2
    cube[:, :, 1] = 4

    result = analyze_data_quality(cube)

    assert np.allclose(result["band_means"], [2, 4])
    assert np.allclose(result["band_stds"], [0, 0])


def test_input_is_not_modified():
    cube = np.random.rand(5, 5, 3).astype(np.float32)
    original = cube.copy()

    analyze_data_quality(cube)

    assert np.array_equal(cube, original)


def test_float64_input():
    cube = np.ones((5, 5, 3), dtype=np.float64)

    result = analyze_data_quality(cube)

    assert result["shape"] == (5, 5, 3)
    assert result["mean"] == 1.0