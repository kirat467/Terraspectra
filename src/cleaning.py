import numpy as np


def bad_band_indices(wavelengths, ranges_nm):
    """Water absorption ranges ke andar aane wale band indices."""
    idx = np.zeros(len(wavelengths), dtype=bool)
    for lo, hi in ranges_nm:
        idx |= (wavelengths >= lo) & (wavelengths <= hi)
    return np.where(idx)[0]


def remove_bands(cube, wavelengths, bad_idx):
    keep = np.setdiff1d(np.arange(cube.shape[0]), bad_idx)
    return cube[keep], wavelengths[keep], keep


def noisy_bands(cube, z=3.0):
    """Band-wise std ke basis par outlier (noisy) bands detect karo."""
    stds = np.nanstd(cube, axis=(1, 2))
    zs = (stds - np.nanmean(stds)) / (np.nanstd(stds) + 1e-8)
    return np.where(np.abs(zs) > z)[0]


def clip_reflectance(cube, scale=10000):
    cube = cube / scale
    return np.clip(cube, 0.0, 1.0)


def fill_nan_spectral(cube):
    """Pixel ke spectrum me chhote NaN gaps ko interpolate karo."""
    b, h, w = cube.shape
    flat = cube.reshape(b, -1)
    x = np.arange(b)
    for i in range(flat.shape[1]):
        col = flat[:, i]
        m = np.isnan(col)
        if m.any() and (~m).sum() > 2:
            col[m] = np.interp(x[m], x[~m], col[~m])
    return flat.reshape(b, h, w)


def mask_from_qa(cube, qa, bad_values=(1, 2, 3)):
    """Cloud/shadow QA mask apply karo (QA band: bad_values = cloud, shadow...)."""
    mask = np.isin(qa, bad_values)
    cube = cube.copy()
    cube[:, mask] = np.nan
    return cube, mask
