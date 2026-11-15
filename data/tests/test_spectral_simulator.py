import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_simulator import (N_BANDS, WAVELENGTHS_NM, band_index, leaf_spectrum, ndvi, red_edge_position,  # noqa: E402
                                soil_spectrum, swir_water_index, water_spectrum)


def test_bands_cover_400_to_2400_nm():
    assert len(WAVELENGTHS_NM) == N_BANDS == 200
    assert WAVELENGTHS_NM[0] == 400 and WAVELENGTHS_NM[-1] == 2400


def test_spectra_are_valid_reflectance():
    s = leaf_spectrum(np.linspace(0, 1, 11))
    assert s.shape == (11, 200)
    assert np.isfinite(s).all() and s.min() >= 0 and s.max() <= 1
    assert leaf_spectrum(0.3).shape == (200,)
    assert soil_spectrum().shape == water_spectrum().shape == (200,)


def test_healthy_leaf_has_the_textbook_shape():
    c = leaf_spectrum(0.0)
    assert c[band_index(550)] > c[band_index(670)]            # green bump above red absorption
    assert c[band_index(800)] > 4 * c[band_index(670)]        # near infrared plateau
    assert c[band_index(1450)] < c[band_index(1300)] * 0.5    # water dip
    assert c[band_index(1940)] < c[band_index(1650)]
    assert 0.78 < ndvi(c) < 0.9


def test_stress_moves_the_red_edge_toward_blue_and_dries_the_leaf():
    healthy, early, severe = leaf_spectrum(0.0), leaf_spectrum(0.25), leaf_spectrum(0.6)
    assert red_edge_position(early) < red_edge_position(healthy) - 2
    assert red_edge_position(severe) < red_edge_position(early)
    assert swir_water_index(healthy) > swir_water_index(early) > swir_water_index(severe)
    assert ndvi(healthy) > ndvi(early) > ndvi(severe)


def test_early_stress_barely_changes_ndvi_but_moves_the_red_edge():
    h, e = leaf_spectrum(0.15), leaf_spectrum(0.0)
    assert abs(ndvi(h) - ndvi(e)) < 0.04
    assert abs(red_edge_position(h) - red_edge_position(e)) > 1.5


def test_brightness_scales_the_spectrum_and_ratios_ignore_it():
    a, b = leaf_spectrum(0.2, brightness=1.0), leaf_spectrum(0.2, brightness=0.8)
    assert np.all(b <= a + 1e-12)
    assert abs(ndvi(a) - ndvi(b)) < 1e-6
