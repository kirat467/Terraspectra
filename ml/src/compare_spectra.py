import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compare_spectra(healthy_cube, diseased_cube, save_path=None):
    healthy_cube = np.asarray(healthy_cube)
    diseased_cube = np.asarray(diseased_cube)

    if healthy_cube.ndim != 3 or diseased_cube.ndim != 3:
        raise ValueError("Both inputs must be 3D cubes")

    if healthy_cube.shape != diseased_cube.shape:
        raise ValueError("Both cubes must have the same shape")

    if not np.isfinite(healthy_cube).all():
        raise ValueError("Healthy cube contains invalid values")

    if not np.isfinite(diseased_cube).all():
        raise ValueError("Diseased cube contains invalid values")

    healthy_mean = healthy_cube.mean(axis=(0, 1))
    diseased_mean = diseased_cube.mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(healthy_mean, label="Healthy")
    ax.plot(diseased_mean, label="Diseased")

    ax.set_title("Healthy vs Diseased Spectral Comparison")
    ax.set_xlabel("Spectral Band Index")
    ax.set_ylabel("Mean Pixel Intensity")
    ax.legend()
    ax.grid(True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return healthy_mean, diseased_mean, fig