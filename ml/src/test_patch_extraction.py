import numpy as np
import pytest

from ml.src.patch_extraction import extract_spatial_patches


def test_patch_extraction():

    cube = np.random.rand(
        140, 280, 550
    ).astype(np.float32)

    patches, labels = extract_spatial_patches(
        cube, label=2
    )

    assert len(patches["train"]) == 16
    assert len(patches["val"]) == 4
    assert len(patches["test"]) == 4

    assert patches["train"][0].shape == (32, 32, 550)
    assert all(x == 2 for x in labels["train"])


def test_invalid_cube():

    with pytest.raises(ValueError):
        extract_spatial_patches(
            np.zeros((32, 32)), 0
        )


def test_invalid_values():

    cube = np.zeros((64, 64, 10))
    cube[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        extract_spatial_patches(cube, 0)