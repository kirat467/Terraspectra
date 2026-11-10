
import numpy as np


def explore_cube(cube):
    """
    Explore a hyperspectral image cube.

    Expected shape: Height, Width, Spectral Bands
    """

    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise ValueError("Image must have 3 dimensions: H, W, Bands")

    height, width, bands = cube.shape

    print("\n--- TerraSpectra Data Exploration ---")
    print("Image Height:", height)
    print("Image Width:", width)
    print("Spectral Bands:", bands)
    print("Data Type:", cube.dtype)
    print("Total Values:", cube.size)
    print("Missing Values:", np.isnan(cube).sum())
    print("Minimum Value:", np.nanmin(cube))
    print("Maximum Value:", np.nanmax(cube))
    print("Mean Value:", np.nanmean(cube))
    print("Standard Deviation:", np.nanstd(cube))

    return {
        "height": height,
        "width": width,
        "bands": bands,
        "dtype": str(cube.dtype)
    }
