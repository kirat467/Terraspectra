import numpy as np
import pytest
import matplotlib.pyplot as plt

from ml.src.compare_spectra import compare_spectra


def test_compare_spectra():
    healthy = np.ones((10, 10, 20))
    diseased = np.full((10, 10, 20), 2.0)

    h, d, fig = compare_spectra(healthy, diseased)

    assert h.shape == (20,)
    assert d.shape == (20,)
    assert np.all(h == 1.0)
    assert np.all(d == 2.0)
    assert fig is not None

    plt.close(fig)


def test_different_shapes():
    healthy = np.random.rand(10, 10, 20)
    diseased = np.random.rand(10, 10, 15)

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)


def test_invalid_dimensions():
    healthy = np.random.rand(10, 10)
    diseased = np.random.rand(10, 10, 5)

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)


def test_empty_cubes():
    healthy = np.empty((0, 10, 5))
    diseased = np.empty((0, 10, 5))

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_invalid_data(invalid_value):
    healthy = np.ones((10, 10, 20))
    diseased = np.ones((10, 10, 20))

    diseased[0, 0, 0] = invalid_value

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)


def test_invalid_healthy_data():
    healthy = np.ones((10, 10, 20))
    diseased = np.ones((10, 10, 20))

    healthy[0, 0, 0] = np.nan

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)


def test_non_numeric_cube():
    healthy = np.full((2, 2, 5), "invalid")
    diseased = np.ones((2, 2, 5))

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)


def test_complex_cube():
    healthy = np.ones((2, 2, 5), dtype=complex)
    diseased = np.ones((2, 2, 5))

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased)


def test_spectral_means():
    healthy = np.ones((4, 4, 3))
    diseased = np.full((4, 4, 3), 5.0)

    h, d, fig = compare_spectra(healthy, diseased)

    np.testing.assert_allclose(h, [1, 1, 1])
    np.testing.assert_allclose(d, [5, 5, 5])

    plt.close(fig)


def test_save_spectrum_plot(tmp_path):
    healthy = np.ones((5, 5, 10))
    diseased = np.full((5, 5, 10), 2.0)

    save_file = tmp_path / "plots" / "spectral.png"

    h, d, fig = compare_spectra(
        healthy,
        diseased,
        save_path=save_file
    )

    assert save_file.exists()
    assert save_file.stat().st_size > 0

    plt.close(fig)


def test_save_path_as_string(tmp_path):
    healthy = np.ones((3, 3, 5))
    diseased = np.full((3, 3, 5), 2.0)

    save_file = str(tmp_path / "spectrum.png")

    h, d, fig = compare_spectra(
        healthy,
        diseased,
        save_path=save_file
    )

    assert (tmp_path / "spectrum.png").exists()

    plt.close(fig)


def test_invalid_save_path_type():
    healthy = np.ones((3, 3, 5))
    diseased = np.ones((3, 3, 5))

    with pytest.raises(TypeError):
        compare_spectra(healthy, diseased, save_path=123)


def test_empty_save_path():
    healthy = np.ones((3, 3, 5))
    diseased = np.ones((3, 3, 5))

    with pytest.raises(ValueError):
        compare_spectra(healthy, diseased, save_path=" ")


def test_plot_contains_both_spectra():
    healthy = np.ones((3, 3, 5))
    diseased = np.full((3, 3, 5), 2.0)

    h, d, fig = compare_spectra(healthy, diseased)

    ax = fig.axes[0]

    assert len(ax.lines) == 2
    assert ax.get_xlabel() == "Spectral Band Index"
    assert ax.get_ylabel() == "Mean Pixel Intensity"

    plt.close(fig)