
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_pca(reduced_cube, save_dir="ml/dataset/processed"):
    reduced_cube = np.asarray(reduced_cube)

    if reduced_cube.ndim != 3:
        raise ValueError("Expected H, W, Components")

    if not np.isfinite(reduced_cube).all():
        raise ValueError("PCA data contains invalid values")

    if reduced_cube.shape[2] < 3:
        raise ValueError("At least 3 PCA components are required")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(3):
        axes[i].imshow(reduced_cube[:, :, i], cmap="viridis")
        axes[i].set_title(f"Principal Component {i+1}")
        axes[i].axis("off")

    plt.tight_layout()

    output_file = save_path / "pca_components.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print("PCA visualization saved:", output_file)
    return str(output_file)
