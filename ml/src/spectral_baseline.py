"""Band-index baselines for disease detection, and a simple full-spectrum detector.

Why this file exists: the plan asks to show that early disease needs more than red and near-infrared light.
It compares, on the simulated scene (data/make_scene.py):
    NDVI                    red against near infrared, the classic "greenness" test
    red edge position       where the steep rise near 720 nm sits (moves toward blue when chlorophyll drops)
    NDII (water index)      860 nm against 1650 nm, drops as the leaf dries
    full spectrum           a ridge regression on all 200 bands
The score for each is AUC: the chance that a randomly picked stressed pixel scores "more diseased" than a randomly
picked healthy pixel. 0.5 is a coin flip, 1.0 is perfect.

`spectral_predict_fn` wraps the full-spectrum detector for risk_map.run(), so the whole pipeline (scan -> risk map ->
dashboard) can be tried before the 3D CNN is trained. Only NumPy is needed.
"""
import numpy as np

WAVELENGTHS_NM = np.linspace(400.0, 2400.0, 200)


def _band(wavelength, wavelengths):
    return int(np.argmin(np.abs(np.asarray(wavelengths) - wavelength)))


def ndvi(cube, wavelengths=WAVELENGTHS_NM):
    red, nir = cube[..., _band(670, wavelengths)], cube[..., _band(800, wavelengths)]
    return (nir - red) / (nir + red + 1e-9)


def red_edge_position(cube, wavelengths=WAVELENGTHS_NM):
    lo, hi = _band(660, wavelengths), _band(780, wavelengths)
    slope = np.clip(np.diff(cube[..., lo:hi + 1], axis=-1), 0, None)
    centres = (np.asarray(wavelengths)[lo:hi] + np.asarray(wavelengths)[lo + 1:hi + 1]) / 2
    return (slope * centres).sum(axis=-1) / (slope.sum(axis=-1) + 1e-12)


def ndii(cube, wavelengths=WAVELENGTHS_NM):
    a, b = cube[..., _band(860, wavelengths)], cube[..., _band(1650, wavelengths)]
    return (a - b) / (a + b + 1e-9)


def disease_score(name, cube, wavelengths=WAVELENGTHS_NM):
    """Higher = more likely diseased, for each index (signs chosen so that stress raises the score)."""
    if name == "ndvi":
        return -ndvi(cube, wavelengths)
    if name == "red_edge":
        return -red_edge_position(cube, wavelengths)
    if name == "ndii":
        return -ndii(cube, wavelengths)
    raise ValueError(f"Unknown index {name}")


def auc(scores_positive, scores_negative):
    """Area under the ROC curve by ranks (Mann-Whitney). Ties get half credit."""
    pos, neg = np.asarray(scores_positive, float), np.asarray(scores_negative, float)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Need at least one positive and one negative pixel")
    both = np.concatenate([pos, neg])
    order = both.argsort(kind="mergesort")
    ranks = np.empty(both.size)
    sorted_vals = both[order]
    i = 0
    while i < both.size:                                    # average ranks over ties
        j = i
        while j + 1 < both.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


class SpectralRidge:
    """Ridge regression from a pixel's 200-band spectrum to a severity in 0..1. Bands are standardised first."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, spectra, severity):
        X = np.asarray(spectra, float)
        y = np.asarray(severity, float)
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        Z = (X - self.mean_) / self.std_
        A = Z.T @ Z + self.alpha * len(Z) * np.eye(Z.shape[1])
        self.coef_ = np.linalg.solve(A, Z.T @ (y - y.mean()))
        self.intercept_ = y.mean()
        return self

    def predict(self, spectra):
        Z = (np.asarray(spectra, float) - self.mean_) / self.std_
        return np.clip(Z @ self.coef_ + self.intercept_, 0.0, 1.0)


def spectral_predict_fn(model):
    """predict_fn for risk_map.run(): patches (N, 1, bands, h, w) -> disease probability per patch.

    The patch score is the 75th percentile of the pixel scores, so a small diseased spot inside a patch still shows."""
    def predict(patches):
        n, _, bands, h, w = patches.shape
        spectra = np.asarray(patches).reshape(n, bands, h * w).transpose(0, 2, 1).reshape(-1, bands)
        pixel = model.predict(spectra).reshape(n, h * w)
        return np.percentile(pixel, 75, axis=1)
    return predict


def compare_detectors(cube, labels, severity, train_cube=None, train_severity=None, train_labels=None, wavelengths=WAVELENGTHS_NM, seed=0):
    """AUC of each detector for early stress (label 1) and visible stress (label 2), each against healthy (label 0).

    The full-spectrum detector is trained on train_* (a different date) so it is not graded on its own training pixels."""
    train_cube = cube if train_cube is None else train_cube
    train_severity = severity if train_severity is None else train_severity
    train_labels = labels if train_labels is None else train_labels
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(train_labels.ravel() >= 0)
    idx = rng.choice(idx, size=min(6000, idx.size), replace=False)
    model = SpectralRidge().fit(train_cube.reshape(-1, train_cube.shape[-1])[idx], train_severity.ravel()[idx])
    flat = cube.reshape(-1, cube.shape[-1])
    lab = labels.ravel()
    scores = {n: disease_score(n, flat, wavelengths) for n in ("ndvi", "red_edge", "ndii")}
    scores["full_spectrum"] = model.predict(flat)
    out = {}
    for name, s in scores.items():
        out[name] = {"early_stress": auc(s[lab == 1], s[lab == 0]) if (lab == 1).any() else None,
                     "visible_stress": auc(s[lab == 2], s[lab == 0]) if (lab == 2).any() else None}
    return out
