
import numpy as np


def validate_hyperspectral_cube(cube):
    """
    Validate a hyperspectral image.

    Expected shape: Height, Width, Spectral Bands
    """

    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError(
            "Image must have 3 dimensions: Height, Width, Bands"
        )

    if any(size == 0 for size in cube.shape):
        raise ValueError("Image cannot have an empty dimension")

    if not np.isfinite(cube).all():
        raise ValueError("Image contains invalid values")

    return cube
