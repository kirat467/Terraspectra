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


def test_invalid_dimensions():
    cube = np.random.rand(10, 10)

    with pytest.raises(ValueError):
        analyze_data_quality(cube)


def test_invalid_values():
    cube = np.ones((10, 10, 5))
    cube[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        analyze_data_quality(cube)