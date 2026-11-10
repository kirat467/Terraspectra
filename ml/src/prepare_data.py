
from pathlib import Path

from .load_data import load_tiff
from .pca import fit_pca, transform_pca
from .visualize_pca import visualize_pca


def prepare_tiff(file_path, n_components=10):
    # Load hyperspectral image
    cube = load_tiff(file_path)

    # Fit PCA on this image for exploratory visualization
    pca = fit_pca(cube, n_components)

    # Reduce spectral bands
    reduced_cube = transform_pca(cube, pca)

    # Save PCA visualization
    output = visualize_pca(reduced_cube)

    print("PCA processing completed!")
    print("Original shape:", cube.shape)
    print("Reduced shape:", reduced_cube.shape)
    print("Visualization:", output)

    return reduced_cube, pca
