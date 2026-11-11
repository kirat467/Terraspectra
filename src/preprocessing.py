import numpy as np
from scipy.signal import savgol_filter


def normalize(cube, method="per_band"):
    if method == "per_band":
        mean = np.nanmean(cube, axis=(1, 2), keepdims=True)
        std = np.nanstd(cube, axis=(1, 2), keepdims=True)
        return (cube - mean) / (std + 1e-8)
    if method == "minmax":
        mn = np.nanmin(cube, axis=(1, 2), keepdims=True)
        mx = np.nanmax(cube, axis=(1, 2), keepdims=True)
        return (cube - mn) / (mx - mn + 1e-8)
    return cube


def smooth_spectra(cube, window=7, polyorder=2):
    """Savitzky-Golay smoothing spectral axis (axis=0) par."""
    filled = np.nan_to_num(cube, nan=0.0)
    out = savgol_filter(filled, window_length=window, polyorder=polyorder, axis=0)
    out[np.isnan(cube)] = np.nan
    return out
