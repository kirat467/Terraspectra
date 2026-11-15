"""Try the whole chain without the 3D CNN: simulated scans -> detector -> risk maps -> dashboard files.

    simulated 200-band scans (data/make_scene.py)
      -> full-spectrum detector (ml/src/spectral_baseline.py, trained on the last scan)
      -> patch scores, 21-day forecast, risk_<date>.json (ml/src/risk_map.py)
      -> frontend/public/data/risk/   (reload the dashboard to see it)

Run from the repository root. Pass --out to write somewhere else and keep the mock files:
    python scripts/demo_end_to_end.py --out /tmp/risk_demo
Swap `spectral_predict_fn(model)` for `torch_predict_fn(cnn)` once the CNN is trained.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "ml" / "src"))
from make_scene import build_scenes  # noqa: E402
from risk_map import run  # noqa: E402
from spectral_baseline import SpectralRidge, spectral_predict_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", default=str(ROOT / "frontend" / "public" / "data"))
    ap.add_argument("--out", default=str(ROOT / "frontend" / "public" / "data" / "risk"))
    ap.add_argument("--patch", type=int, default=9)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()

    meta, geo, owner, scenes = build_scenes(a.farm)
    train = scenes[-1]
    # train on every pixel of the last scan: tracks, ponds and fallow soil count as "not diseased" (severity 0)
    target = np.where(owner.ravel() >= 0, train["severity"].ravel(), 0.0)
    rng = np.random.default_rng(0)
    pick = rng.choice(target.size, size=8000, replace=False)
    model = SpectralRidge().fit(train["cube"].reshape(-1, 200)[pick], target[pick])

    index = run([(s["date"], s["cube"]) for s in scenes], spectral_predict_fn(model), meta, geo, a.out, patch_size=a.patch, stride=a.stride)
    print(f"wrote {len(index['dates'])} risk maps to {a.out}")
    for h in index["history"]:
        print(f"  {h['date']}: {h['acres_high']:5.1f} ac high risk, {h['acres_moderate']:5.1f} ac moderate, {h['zones']} zone(s)")
    if "simulated" in meta:
        print("Note: the data is simulated, so these maps show the pipeline working, not real disease.")


if __name__ == "__main__":
    main()
