
import pytest

from ml.src.load_data import load_tiff, inspect_hdf5


def test_tiff_missing_file():
    with pytest.raises(FileNotFoundError):
        load_tiff("ml/dataset/raw/missing.tif")


def test_hdf5_missing_file():
    with pytest.raises(FileNotFoundError):
        inspect_hdf5("ml/dataset/raw/missing.h5")
