import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_spectral_signature(cube, row, col, save_path=None):
    cube = np.asarray(cube)

    if cube.ndim != 3 or cube.size == 0:
        raise ValueError("Invalid hyperspectral cube")

    height, width, bands = cube.shape

    if not (0 <= row < height and 0 <= col < width):
        raise ValueError("Pixel coordinates are out of range")

    signature = cube[row, col, :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(bands), signature)

    ax.set_title("Tomato Leaf Spectral Signature")
    ax.set_xlabel("Spectral Band Index")
    ax.set_ylabel("Pixel Intensity")
    ax.grid(True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig


def plot_band_images(cube, bands=(50, 250, 450), save_path=None):
    cube = np.asarray(cube)

    if cube.ndim != 3 or cube.size == 0:
        raise ValueError("Invalid hyperspectral cube")

    if any(b < 0 or b >= cube.shape[2] for b in bands):
        raise ValueError("Invalid spectral band index")

    fig, axes = plt.subplots(1, len(bands), figsize=(12, 4))

    if len(bands) == 1:
        axes = [axes]

    for ax, band in zip(axes, bands):
        ax.imshow(cube[:, :, band], cmap="gray")
        ax.set_title(f"Band {band}")
        ax.axis("off")

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig