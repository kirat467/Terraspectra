
import numpy as np


def validate_hyperspectral_cube(cube):
    """
    Validate a hyperspectral image cube.

    Expected shape: Height, Width, Spectral Bands
    """

    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise ValueError("Image must have 3 dimensions: H, W, Bands")

    if any(size == 0 for size in cube.shape):
        raise ValueError("Image cannot have an empty dimension")

    if not np.issubdtype(cube.dtype, np.number):
        raise ValueError("Image must contain numeric values")

    if not np.isfinite(cube).all():
        raise ValueError("Image contains NaN or infinite values")

    cube = cube.astype(np.float32)

    print("Hyperspectral image validation successful!")
    print("Image shape:", cube.shape)
    print("Data type:", cube.dtype)

    return cube
