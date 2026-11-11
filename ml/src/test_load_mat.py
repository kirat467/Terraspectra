import numpy as np
import pytest
import scipy.io
from ml.src.load_mat import inspect_mat_file, load_mat_cube


def test_missing_mat_file():
    with pytest.raises(FileNotFoundError):
        inspect_mat_file("missing_dataset.mat")


def test_load_mat_cube(tmp_path):
    cube = np.random.rand(10, 12, 20)
    file_path = tmp_path / "sample.mat"

    scipy.io.savemat(file_path, {"Day_one_1": cube})

    result = load_mat_cube(file_path)

    assert result.shape == (10, 12, 20)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_missing_variable(tmp_path):
    file_path = tmp_path / "sample.mat"
    scipy.io.savemat(file_path, {"Other_variable": np.ones((5, 5, 5))})

    with pytest.raises(KeyError):
        load_mat_cube(file_path)