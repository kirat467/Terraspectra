import numpy as np


def _band(cube, wavelengths, nm):
    return cube[int(np.argmin(np.abs(wavelengths - nm)))]


def ndvi(cube, wl):
    nir, red = _band(cube, wl, 800), _band(cube, wl, 670)
    return (nir - red) / (nir + red + 1e-8)


def pri(cube, wl):
    b531, b570 = _band(cube, wl, 531), _band(cube, wl, 570)
    return (b531 - b570) / (b531 + b570 + 1e-8)


def red_edge_ndvi(cube, wl):
    b750, b705 = _band(cube, wl, 750), _band(cube, wl, 705)
    return (b750 - b705) / (b750 + b705 + 1e-8)


def evi(cube, wl):
    nir, red, blue = _band(cube, wl, 800), _band(cube, wl, 670), _band(cube, wl, 480)
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)


def all_indices(cube, wl):
    return {
        "NDVI": ndvi(cube, wl),
        "PRI": pri(cube, wl),
        "RE_NDVI": red_edge_ndvi(cube, wl),
        "EVI": evi(cube, wl),
    }
