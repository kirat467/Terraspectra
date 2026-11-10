
from pathlib import Path

from .load_data import load_tiff
from .pca import fit_pca, transform_pca
from .visualize_pca import visualize_pca


def prepare_tiff(file_path, n_components=10):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # Load hyperspectral image
    cube = load_tiff(file_path)

    # Validate PCA components
    if n_components < 3:
        raise ValueError("At least 3 PCA components are required")

    if n_components > min(cube.shape):
        raise ValueError("Too many PCA components")

    # Fit PCA
    pca = fit_pca(cube, n_components)

    # Reduce spectral bands
    reduced_cube = transform_pca(cube, pca)

    # Save PCA visualization
    output = visualize_pca(reduced_cube)

    print("\n--- TerraSpectra PCA Processing ---")
    print("Original shape:", cube.shape)
    print("Reduced shape:", reduced_cube.shape)
    print("PCA components:", n_components)
    print("Visualization saved at:", output)
    print("PCA processing completed successfully!")

    return reduced_cube, pca
