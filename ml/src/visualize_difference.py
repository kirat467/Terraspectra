import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_spectral_difference(difference, top_bands, save_path=None):
    difference = np.asarray(difference)
    top_bands = np.asarray(top_bands)

    if difference.ndim != 1 or difference.size == 0:
        raise ValueError("Difference must be a non-empty 1D array")

    if not np.isfinite(difference).all():
        raise ValueError("Difference contains invalid values")

    if top_bands.ndim != 1 or top_bands.size == 0:
        raise ValueError("Top bands must be a non-empty 1D array")

    if not np.issubdtype(top_bands.dtype, np.integer):
        raise ValueError("Band indices must be integers")

    if np.any(top_bands < 0) or np.any(top_bands >= len(difference)):
        raise ValueError("Invalid spectral band index")

    fig, ax = plt.subplots(figsize=(12, 5))

    bands = np.arange(len(difference))

    ax.plot(bands, difference, label="Spectral Difference")
    ax.scatter(
        top_bands,
        difference[top_bands],
        label="Top Different Bands"
    )

    ax.set_title("Healthy vs Diseased Spectral Difference")
    ax.set_xlabel("Spectral Band Index")
    ax.set_ylabel("Absolute Mean Difference")
    ax.legend()
    ax.grid(True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    return fig