
import numpy as np
import pytest

from explore_data import explore_cube


def test_explore_cube():

    cube = np.random.rand(20, 20, 200)

    report = explore_cube(cube)

    assert report["height"] == 20
    assert report["width"] == 20
    assert report["bands"] == 200
    assert report["total_values"] == 80000


def test_invalid_cube():

    with pytest.raises(ValueError):
        explore_cube(np.random.rand(20, 20))


def test_empty_cube():

    with pytest.raises(ValueError):
        explore_cube(np.empty((0, 20, 200)))


def test_cube_with_missing_values():

    cube = np.random.rand(10, 10, 5)
    cube[0, 0, 0] = np.nan

    report = explore_cube(cube)

    assert report["missing_values"] == 1
