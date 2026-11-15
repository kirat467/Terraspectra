import sys
from pathlib import Path

import numpy as np
import pytest

from spectral_baseline import SpectralRidge, auc, compare_detectors, disease_score, spectral_predict_fn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
from make_scene import build_scenes  # noqa: E402


def test_auc_known_values():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    assert auc([1, 3], [2]) == 0.5
    with pytest.raises(ValueError):
        auc([], [1])


def test_ridge_recovers_a_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 10))
    y = np.clip(0.5 + 0.2 * X[:, 0] - 0.1 * X[:, 3], 0, 1)
    m = SpectralRidge(alpha=1e-4).fit(X, y)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.98
    assert m.predict(X).min() >= 0 and m.predict(X).max() <= 1


def test_unknown_index_is_rejected():
    with pytest.raises(ValueError):
        disease_score("evi", np.zeros((2, 200)))


def test_predict_fn_returns_one_probability_per_patch():
    rng = np.random.default_rng(1)
    model = SpectralRidge().fit(rng.random((200, 12)), rng.random(200))
    out = spectral_predict_fn(model)(rng.random((5, 1, 12, 4, 4)))
    assert out.shape == (5,) and (0 <= out).all() and (out <= 1).all()


def test_full_spectrum_beats_ndvi_for_early_stress():
    _, _, _, sc = build_scenes(ROOT / "frontend" / "public" / "data")
    today, train = sc[2], sc[5]
    r = compare_detectors(today["cube"], today["label"], today["severity"], train["cube"], train["severity"], train["label"])
    assert r["full_spectrum"]["early_stress"] > 0.95
    assert r["full_spectrum"]["early_stress"] > r["red_edge"]["early_stress"] > r["ndvi"]["early_stress"]
