import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compare_spectra(healthy_cube, diseased_cube, save_path=None):
    try:
        healthy_cube = np.asarray(healthy_cube)
        diseased_cube = np.asarray(diseased_cube)
    except (TypeError, ValueError) as exc:
        raise ValueError("Unable to convert input cubes to arrays") from exc

    if healthy_cube.ndim != 3 or diseased_cube.ndim != 3:
        raise ValueError("Both inputs must be 3D cubes")

    if healthy_cube.size == 0 or diseased_cube.size == 0:
        raise ValueError("Both cubes must contain data")

    if healthy_cube.shape != diseased_cube.shape:
        raise ValueError("Both cubes must have the same shape")

    for name, cube in (
        ("Healthy", healthy_cube),
        ("Diseased", diseased_cube)
    ):
        if (
            not np.issubdtype(cube.dtype, np.number)
            or np.issubdtype(cube.dtype, np.complexfloating)
        ):
            raise ValueError(f"{name} cube must contain real numeric values")

        if not np.isfinite(cube).all():
            raise ValueError(f"{name} cube contains invalid values")

    healthy_mean = healthy_cube.mean(axis=(0, 1))
    diseased_mean = diseased_cube.mean(axis=(0, 1))

    if not np.isfinite(healthy_mean).all():
        raise ValueError("Healthy spectral mean contains invalid values")

    if not np.isfinite(diseased_mean).all():
        raise ValueError("Diseased spectral mean contains invalid values")

    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        bands = np.arange(healthy_mean.size)

        ax.plot(bands, healthy_mean, label="Healthy")
        ax.plot(bands, diseased_mean, label="Diseased")

        ax.set_title("Healthy vs Diseased Spectral Comparison")
        ax.set_xlabel("Spectral Band Index")
        ax.set_ylabel("Mean Pixel Intensity")
        ax.legend()
        ax.grid(True)

        if save_path is not None:
            if not isinstance(save_path, (str, Path)):
                raise TypeError("Save path must be a string or Path object")

            save_path = Path(save_path)

            if not str(save_path).strip():
                raise ValueError("Save path cannot be empty")

            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)

    except Exception:
        plt.close(fig)
        raise

    return healthy_mean, diseased_mean, fig