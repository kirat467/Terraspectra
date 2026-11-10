
import numpy as np
from sklearn.decomposition import PCA

def reduce_spectral_bands(cube, n_components):
    """
    Reduce hyperspectral spectral bands using PCA.

    Input shape: Height, Width, Bands
    """

    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("Input must have 3 dimensions: H, W, Bands")

    if not np.isfinite(cube).all():
        raise ValueError("Cube contains invalid values")

    height, width, bands = cube.shape

    if not 1 <= n_components <= min(height * width, bands):
        raise ValueError("Invalid number of PCA components")

    pixels = cube.reshape(-1, bands)

    pca = PCA(n_components=n_components)

    reduced = pca.fit_transform(pixels)

    reduced_cube = reduced.reshape(
        height, width, n_components
    )

    return reduced_cube, pca
