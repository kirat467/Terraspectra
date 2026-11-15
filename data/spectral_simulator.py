"""Simulated reflectance spectra for a crop under increasing disease stress.

The real hyperspectral dataset is still pending, so this module builds spectra from the physics a hyperspectral
sensor sees in leaves. It is a teaching model, not a measured one, and every number below is a documented choice.

Wavelengths: 200 bands from 400 to 2400 nm (about 10 nm apart).

What a healthy leaf looks like (reflectance 0 to 1):
    400-700 nm   low, with a small bump at 550 nm (green). Chlorophyll absorbs blue and red.
    ~700-750 nm  the red edge: a steep rise. Its position moves toward blue when chlorophyll drops.
    750-1300 nm  high plateau (leaf structure scatters near infrared light).
    1450, 1940 nm  deep dips from water. 970 and 1200 nm  small dips.
    1300-2400 nm  falls off, with a bump near 1650 and 2200 nm.

What stress does, as severity s goes from 0 (healthy) to 1 (severe):
    chlorophyll falls  -> red reflectance rises, red edge shifts blue (up to 18 nm)
    leaf water falls   -> water dips get shallower, shortwave infrared (SWIR) reflectance rises
    leaf structure collapses only later (s > 0.4) -> near infrared plateau drops
So at low severity the first signs are in the red edge and SWIR, while NDVI (red against near infrared) barely
moves. That is the reason to use many bands, and it is what ml/src/spectral_baseline.py measures.
"""
import numpy as np

N_BANDS = 200
WAVELENGTHS_NM = np.linspace(400.0, 2400.0, N_BANDS)

EARLY_STRESS = 0.08       # severity where the first spectral change starts
VISIBLE_STRESS = 0.45     # severity where symptoms can be seen by eye


def gauss(x, centre, width):
    return np.exp(-0.5 * ((x - centre) / width) ** 2)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def leaf_spectrum(severity, nir_scale=1.0, brightness=1.0, wavelengths=WAVELENGTHS_NM):
    """Reflectance of a crop canopy. severity, nir_scale, brightness are scalars or arrays of shape (N,).

    Returns an array (N, bands) for array input, or (bands,) for scalars.
    """
    s = np.atleast_1d(np.asarray(severity, dtype=np.float64))[:, None]
    scale = np.broadcast_to(np.asarray(nir_scale, dtype=np.float64), s.shape[:1])[:, None]
    bright = np.broadcast_to(np.asarray(brightness, dtype=np.float64), s.shape[:1])[:, None]
    wl = wavelengths[None, :]

    sc = np.clip(s, 0, 1)
    chlorophyll = 1.0 - 0.75 * sc ** 2                                  # visible colour only changes once the loss is large
    water = 1.0 - 0.55 * sc
    structure = 1.0 - 0.45 * np.clip((s - 0.4) / 0.6, 0, 1) ** 1.2        # only drops late

    # visible: blue and red absorption weaker when chlorophyll is low, green bump
    absorb = 0.9 * chlorophyll
    visible = 0.022 + 0.075 * gauss(wl, 550, 35) + 0.30 * (1 - absorb) * (gauss(wl, 480, 60) + gauss(wl, 670, 40)) / 1.6 \
        + 0.07 * (1 - chlorophyll) * sigmoid((wl - 560) / 20)

    # red edge position moves toward blue with stress
    edge = 718.0 - 18.0 * sc                                 # shifts early, in step with severity
    step = sigmoid((wl - edge) / 11.0)

    # near infrared plateau and SWIR shoulder
    nir = (0.50 - 0.10 * (wl - 800) / 1600.0) * scale * structure
    nir = nir * np.where(wl > 1300, np.exp(-(wl - 1300) / 900.0 * 0.9), 1.0)
    swir_stress = 0.09 * np.clip(s, 0, 1) * sigmoid((wl - 1300) / 60.0)           # drier leaves reflect more SWIR

    base = visible * (1 - step) + (nir + 0.0) * step + swir_stress
    # water absorption dips (shallower when the leaf is drier)
    for centre, width, depth in [(970, 22, 0.10), (1200, 28, 0.14), (1450, 55, 0.60), (1940, 70, 0.80)]:
        base = base * (1 - depth * water * gauss(wl, centre, width))
    out = np.clip(base * bright, 0.0, 1.0)
    return out if np.ndim(severity) else out[0]


def soil_spectrum(brightness=1.0, wavelengths=WAVELENGTHS_NM):
    """Dry bare soil: smooth rise from visible to SWIR with weak water dips."""
    wl = wavelengths
    base = 0.10 + 0.28 * (wl - 400) / 2000.0 + 0.03 * gauss(wl, 2200, 60)
    for centre, width, depth in [(1450, 55, 0.12), (1940, 70, 0.18)]:
        base = base * (1 - depth * gauss(wl, centre, width))
    return np.clip(base * brightness, 0, 1)


def water_spectrum(wavelengths=WAVELENGTHS_NM):
    """Pond water: dark, a little blue-green light, nothing beyond 900 nm."""
    wl = wavelengths
    return np.clip(0.06 * gauss(wl, 520, 90) + 0.012 + 0.0, 0, 1) * np.where(wl > 900, 0.3, 1.0)


# ---- indices used by the baseline and the tests (plain band maths on any (..., bands) array)
def band_index(wavelength_nm, wavelengths=WAVELENGTHS_NM):
    return int(np.argmin(np.abs(wavelengths - wavelength_nm)))


def ndvi(cube, wavelengths=WAVELENGTHS_NM):
    red, nir = cube[..., band_index(670, wavelengths)], cube[..., band_index(800, wavelengths)]
    return (nir - red) / (nir + red + 1e-9)


def red_edge_position(cube, wavelengths=WAVELENGTHS_NM):
    """Red edge position in nm: the slope-weighted average wavelength between 660 and 780 nm."""
    lo, hi = band_index(660, wavelengths), band_index(780, wavelengths)
    slope = np.clip(np.diff(cube[..., lo:hi + 1], axis=-1), 0, None)
    centres = (wavelengths[lo:hi] + wavelengths[lo + 1:hi + 1]) / 2
    return (slope * centres).sum(axis=-1) / (slope.sum(axis=-1) + 1e-12)


def swir_water_index(cube, wavelengths=WAVELENGTHS_NM):
    """Normalised difference of 860 nm and 1650 nm: rises as leaf water falls (NDII)."""
    a, b = cube[..., band_index(860, wavelengths)], cube[..., band_index(1650, wavelengths)]
    return (a - b) / (a + b + 1e-9)
