import numpy as np
import pytest

from class_spectral_quality import compare_class_spectral_quality


def test_class_spectral_quality():
    cubes = {
        "Healthy": np.random.rand(10, 10, 5),
        "Diseased": np.random.rand(10, 10, 5)
    }

    result = compare_class_spectral_quality(cubes)

    assert len(result) == 2
    assert "Healthy" in result
    assert "Diseased" in result
    assert len(result["Healthy"]["band_means"]) == 5


def test_empty_cubes():
    with pytest.raises(ValueError):
        compare_class_spectral_quality({})


def test_invalid_cube():
    cubes = {
        "Healthy": np.random.rand(10, 10)
    }

    with pytest.raises(ValueError):
        compare_class_spectral_quality(cubes)


def test_invalid_values():
    cube = np.ones((10, 10, 5))
    cube[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        compare_class_spectral_quality({"Healthy": cube})