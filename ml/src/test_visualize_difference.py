import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from visualize_difference import plot_spectral_difference


def test_plot_spectral_difference():
    difference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    top_bands = np.array([4, 3, 2])

    fig = plot_spectral_difference(difference, top_bands)

    assert fig is not None
    assert len(fig.axes) == 1


def test_save_graph(tmp_path):
    difference = np.array([0.1, 0.2, 0.3])
    top_bands = np.array([2, 1])

    path = tmp_path / "difference.png"

    fig = plot_spectral_difference(difference, top_bands, path)

    assert path.exists()
    assert path.stat().st_size > 0

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_invalid_difference():
    with pytest.raises(ValueError):
        plot_spectral_difference(np.array([]), np.array([0]))


def test_invalid_band():
    difference = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        plot_spectral_difference(difference, np.array([5]))


def test_invalid_values():
    difference = np.array([0.1, np.nan, 0.3])

    with pytest.raises(ValueError):
        plot_spectral_difference(difference, np.array([0]))