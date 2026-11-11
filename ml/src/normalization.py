import numpy as np


def normalize_spectral_cube(cube, method="zscore"):
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3 or cube.size == 0:
        raise ValueError("Cube must be a non-empty 3D array")

    if not np.isfinite(cube).all():
        raise ValueError("Cube contains invalid values")

    if method not in ("zscore", "minmax"):
        raise ValueError("Invalid normalization method")

    if method == "zscore":
        mean = cube.mean(axis=(0, 1), keepdims=True)
        std = cube.std(axis=(0, 1), keepdims=True)
        std[std == 0] = 1
        return (cube - mean) / std

    minimum = cube.min(axis=(0, 1), keepdims=True)
    maximum = cube.max(axis=(0, 1), keepdims=True)
    difference = maximum - minimum
    difference[difference == 0] = 1

    return (cube - minimum) / difference