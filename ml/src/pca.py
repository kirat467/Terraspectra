
import numpy as np
from sklearn.decomposition import PCA


def validate_cube(cube):
    cube = np.asarray(cube, dtype=np.float32)

    if cube.ndim != 3:
        raise ValueError("Input must have 3 dimensions: H, W, Bands")

    if not np.isfinite(cube).all():
        raise ValueError("Cube contains invalid values")

    return cube


def fit_pca(train_cube, n_components=10):
    train_cube = validate_cube(train_cube)

    height, width, bands = train_cube.shape

    if not 1 <= n_components <= min(height * width, bands):
        raise ValueError("Invalid number of PCA components")

    pixels = train_cube.reshape(-1, bands)

    pca = PCA(n_components=n_components)
    pca.fit(pixels)

    return pca


def transform_pca(cube, pca):
    cube = validate_cube(cube)

    if cube.shape[2] != pca.n_features_in_:
        raise ValueError("Spectral band count does not match PCA")

    height, width, bands = cube.shape

    pixels = cube.reshape(-1, bands)
    reduced = pca.transform(pixels)

    return reduced.reshape(height, width, -1)


def reduce_spectral_bands(cube, n_components=10):
    pca = fit_pca(cube, n_components)
    reduced_cube = transform_pca(cube, pca)

    return reduced_cube, pca
