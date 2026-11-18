"""Run the trained 3D-CNN over the six simulated scans and write the dashboard's risk files.

    scenes (200 bands) -> PCA (fitted in make_patches.py) -> standardise (train statistics) -> 3D-CNN on 9 x 9 x 20 patches
    -> disease probability (1 - P(healthy)) -> 21-day trend forecast -> risk_<date>.json   (ml/src/risk_map.py)

Needs the files made by data/make_scene.py, data/make_patches.py and ml/src/train_on_scenes.py.
By default it writes to /tmp-style folder next to the data so the mock files stay untouched; pass
--out frontend/public/data/risk to show the CNN output in the dashboard.

    python scripts/cnn_to_dashboard.py --out frontend/public/data/risk_cnn
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml" / "src"))
sys.path.insert(0, str(ROOT / "data"))
from load_scene import list_dates, load_scene  # noqa: E402
from model_3dcnn import Crop3DCNN  # noqa: E402
from risk_map import run, torch_predict_fn  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenes", default=str(ROOT / "data" / "generated" / "scenes.h5"))
    ap.add_argument("--patches", default=str(ROOT / "data" / "generated" / "patches.npz"))
    ap.add_argument("--run", default=str(ROOT / "ml" / "models" / "cnn_run"))
    ap.add_argument("--farm", default=str(ROOT / "frontend" / "public" / "data"))
    ap.add_argument("--out", default=str(ROOT / "frontend" / "public" / "data" / "risk_cnn"))
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    import torch

    z = np.load(a.patches)
    pca_mean, pca_components = z["pca_mean"], z["pca_components"]
    norm = np.load(Path(a.run) / "normalization.npz")
    mean, std = norm["mean"].reshape(-1), norm["std"].reshape(-1)
    model = Crop3DCNN(num_classes=3)
    model.load_state_dict(torch.load(Path(a.run) / "model.pth", map_location="cpu"))
    disease = torch_predict_fn(model)

    def predict(patches):                      # patches arrive as (N, 1, 20, 9, 9) of PCA values
        x = (patches - mean[None, None, :, None, None]) / std[None, None, :, None, None]
        return disease(x.astype(np.float32))

    scans = []
    for d in list_dates(a.scenes):
        cube = load_scene(a.scenes, d)["cube"]
        scans.append((d, ((cube - pca_mean) @ pca_components.T).astype(np.float32)))
    meta = json.loads((Path(a.farm) / "farm_meta.json").read_text())
    geo = json.loads((Path(a.farm) / "farm_fields.geojson").read_text())
    index = run(scans, predict, meta, geo, a.out, patch_size=9, stride=a.stride)
    print(f"wrote {len(index['dates'])} risk maps to {a.out}")
    for h in index["history"]:
        print(f"  {h['date']}: {h['acres_high']:5.1f} ac high, {h['acres_moderate']:5.1f} ac moderate, {h['zones']} zone(s)")


if __name__ == "__main__":
    main()
