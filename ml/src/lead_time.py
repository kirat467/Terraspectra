"""Lead-time analysis: how many days BEFORE symptoms become visible does each detector flag stress?

For every pixel that reaches visible stress (label 2), take the first scan where it is visible and the first scan where a detector
flags it. Lead time = days between the two. Positive means flagged early, zero the same scan, negative flagged only after the
symptoms were already visible, and "never" means missed.

Fairness rules:
  * Everything is scored on a NEW simulated farm made by data/make_large_scene.py (different field layout, crops, outbreak places and
    noise from the farm the CNN was trained on). The original test blocks hold a single outbreak, centred on a pond, which is too
    little to measure lead time (79 labelled pixels, most of them water), so they are not used here.
  * Every detector gets the same false-alarm budget: its threshold is set so that only `fpr` (default 1 percent) of the healthy
    pixels in the TRAIN and VALIDATION blocks are flagged in a single scan.
  * Pixels that are not crops or water, and pixels within 4 pixels of them, are left out for every detector.
  * The band indices (NDVI, red edge position, NDII) are averaged over 5 x 5 pixels first, to match the CNN's 9 x 9 view.

Scans are a week apart, so lead times come in steps of 7 days. A pixel already flagged on the very first scan has a lead time that
is only a lower bound (we do not know when it would first have been flagged); the report counts those separately.

    python ml/src/lead_time.py                       # needs data/generated/patches.npz and ml/models/cnn_run/ (from make_patches.py and train_on_scenes.py)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))
sys.path.insert(0, str(ROOT / "data"))

BINS = [("21+ days before", lambda d: d >= 21), ("14 days before", lambda d: d == 14), ("7 days before", lambda d: d == 7),
        ("same scan", lambda d: d == 0), ("after visible", lambda d: d < 0)]


def box_mean(a, size):
    """Mean over a size x size window for every (H, W) layer of a (T, H, W) array (edges use what is available)."""
    from scipy.ndimage import uniform_filter

    return np.stack([uniform_filter(x, size=size, mode="nearest") for x in a])


def first_true(hits):
    """Index of the first True along axis 0 for each pixel, -1 where there is none. hits: (T, H, W) bool."""
    return np.where(hits.any(axis=0), hits.argmax(axis=0), -1)


def threshold_for_fpr(healthy_scores, fpr):
    """Score above which only `fpr` of the healthy pixels fall (per scan)."""
    return float(np.quantile(healthy_scores, 1.0 - fpr))


def lead_days(first_detect, first_visible, days):
    """Per pixel: days between first detection and first visible stress (nan where never detected). days: day number per scan."""
    days = np.asarray(days, dtype=float)
    lead = np.full(first_detect.shape, np.nan)
    ok = first_detect >= 0
    lead[ok] = days[first_visible[ok]] - days[first_detect[ok]]
    return lead


def evaluate(scores, labels, days, mask, calib_mask, fpr=0.01):
    """Lead-time summary for one detector. scores, labels: (T, H, W). mask: pixels to score. calib_mask: pixels used to set the threshold."""
    crop = labels >= 0
    never_stressed = (labels.max(axis=0) == 0) & crop.all(axis=0)
    calib = calib_mask & never_stressed
    thr = threshold_for_fpr(scores[:, calib].ravel(), fpr)
    first_detect = first_true(scores >= thr)
    visible = labels == 2
    first_visible = first_true(visible)
    target = mask & visible.any(axis=0)
    lead = lead_days(first_detect[target], first_visible[target], days)
    censored = first_detect[target] == 0
    healthy = mask & never_stressed
    out = {"threshold": thr, "pixels_reaching_visible": int(target.sum()), "never_flagged": float(np.isnan(lead).mean()),
           "false_alarm_ever": float((first_detect[healthy] >= 0).mean()), "flagged_on_first_scan": float(censored.mean()),
           "median_lead_days": float(np.nanmedian(lead)) if np.isfinite(lead).any() else None,
           "mean_lead_days": float(np.nanmean(lead)) if np.isfinite(lead).any() else None}
    shares = {}
    for name, rule in BINS:
        shares[name] = float(np.mean([(not np.isnan(v)) and rule(v) for v in lead]))
    shares["never flagged"] = out["never_flagged"]
    out["shares"] = shares
    return out


def ground_truth_lead(labels, days, mask):
    """The simulation's own head start: first scan with any stress (label >= 1) to first scan with visible stress."""
    first_any = first_true(labels >= 1)
    first_vis = first_true(labels == 2)
    target = mask & (first_vis >= 0)
    lead = lead_days(first_any[target], first_vis[target], days)
    return {"pixels": int(target.sum()), "median_lead_days": float(np.nanmedian(lead)), "mean_lead_days": float(np.nanmean(lead))}


def chart(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["methods"])
    cats = [n for n, _ in BINS] + ["never flagged"]
    colours = ["#14407a", "#2a78d6", "#7fb1ea", "#eda100", "#c9ccc8", "#eb6834"]
    fig, ax = plt.subplots(figsize=(10, 0.9 * len(names) + 2.2), dpi=150)
    for y, n in enumerate(names):
        left = 0.0
        for cat, col in zip(cats, colours):
            v = report["methods"][n]["shares"][cat] * 100
            ax.barh(y, v, left=left, color=col, edgecolor="#fcfcfb", linewidth=2, height=0.62)
            if v >= 7:
                ax.text(left + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white" if col in ("#14407a", "#2a78d6", "#eb6834") else "#26302a")
            left += v
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of pixels that reach visible stress (3 new simulated farms, 6 weekly scans each)")
    ax.set_title("How early each detector flags stress, relative to the first scan with visible symptoms", fontsize=11, loc="left")
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours]
    ax.legend(handles, cats, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)


def run(patches, model_dir, out_json, out_png, fpr=0.01, stride=2, size=192, margin=4, farms=3):
    """Score every detector on NEW simulated farms (make_large_scene.py: other field layouts, crops, outbreak places and noise)."""
    import tempfile

    import torch
    from model_3dcnn import Crop3DCNN
    from risk_map import scan_disease_map, torch_predict_fn
    from scipy.ndimage import binary_dilation
    from spectral_baseline import disease_score
    from spectral_simulator import WAVELENGTHS_NM as wl
    import make_large_scene as mls

    z = np.load(patches)
    norm = np.load(Path(model_dir) / "normalization.npz")
    mean, std = norm["mean"].reshape(-1), norm["std"].reshape(-1)
    model = Crop3DCNN(num_classes=3)
    model.load_state_dict(torch.load(Path(model_dir) / "model.pth", map_location="cpu"))
    disease = torch_predict_fn(model)
    predict = lambda p: disease(((p - mean[None, None, :, None, None]) / std[None, None, :, None, None]).astype(np.float32))
    days = [7 * i for i in range(6)]
    per_farm = []
    for farm in range(farms):
        cubes, labels = [], []
        with tempfile.TemporaryDirectory() as tmp:
            for step in range(6):
                mls.generate(tmp, size=size, step=step, strip=32, farm=farm)
                cubes.append(np.load(Path(tmp) / f"large_cube_{step}.npy").astype(np.float32) / mls.SCALE)
                labels.append(np.load(Path(tmp) / f"large_labels_{step}.npy"))
        labels = np.stack(labels)
        mask = ~binary_dilation((labels < 0).any(axis=0), iterations=margin)         # leave out pixels near soil, tracks and water
        scores = {}
        for name, key in [("NDVI", "ndvi"), ("red edge position", "red_edge"), ("NDII (leaf water)", "ndii")]:
            scores[name] = box_mean(np.stack([disease_score(key, c, wl) for c in cubes]), 5)
        scores["3D-CNN"] = np.stack([scan_disease_map(((c - z["pca_mean"]) @ z["pca_components"].T).astype(np.float32), predict, patch_size=9, stride=stride) for c in cubes])
        per_farm.append({"ground_truth": ground_truth_lead(labels, days, mask), "methods": {n: evaluate(s, labels, days, mask, mask, fpr) for n, s in scores.items()}})

    report = {"evaluation": f"{farms} new simulated farms, {size} x {size} pixels, 6 weekly scans each", "fpr_per_scan": fpr, "per_farm": per_farm, "methods": {}}
    for name in per_farm[0]["methods"]:
        parts = [f["methods"][name] for f in per_farm]
        w = np.array([p["pixels_reaching_visible"] for p in parts], dtype=float)
        pooled = {c: float(np.average([p["shares"][c] for p in parts], weights=w)) for c in parts[0]["shares"]}
        report["methods"][name] = {"shares": pooled, "pixels_reaching_visible": int(w.sum()),
                                   "median_lead_days_per_farm": [p["median_lead_days"] for p in parts],
                                   "flagged_on_first_scan": float(np.average([p["flagged_on_first_scan"] for p in parts], weights=w)),
                                   "false_alarm_ever": float(np.mean([p["false_alarm_ever"] for p in parts])),
                                   "share_at_least_14_days_per_farm": [round(p["shares"]["21+ days before"] + p["shares"]["14 days before"], 3) for p in parts]}
    report["ground_truth_median_lead_days_per_farm"] = [f["ground_truth"]["median_lead_days"] for f in per_farm]
    Path(out_json).write_text(json.dumps(report, indent=2))
    chart(report, out_png)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patches", default=str(ROOT / "data" / "generated" / "patches.npz"))
    ap.add_argument("--run", default=str(ROOT / "ml" / "models" / "cnn_run"))
    ap.add_argument("--out-json", default=str(ROOT / "docs" / "lead_time.json"))
    ap.add_argument("--out-png", default=str(ROOT / "docs" / "figures" / "lead_time.png"))
    ap.add_argument("--fpr", type=float, default=0.01)
    a = ap.parse_args()
    Path(a.out_png).parent.mkdir(parents=True, exist_ok=True)
    r = run(a.patches, a.run, a.out_json, a.out_png, a.fpr)
    print("simulation's own head start (median days, per farm):", r["ground_truth_median_lead_days_per_farm"])
    for n, m in r["methods"].items():
        print(f"{n:20s} pixels {m['pixels_reaching_visible']}, median lead per farm {m['median_lead_days_per_farm']}, at least 14 d per farm {m['share_at_least_14_days_per_farm']}, "
              f"first-scan {m['flagged_on_first_scan']:.0%}, false alarms ever {m['false_alarm_ever']:.1%}\n   shares {({k: round(v, 2) for k, v in m['shares'].items()})}")


if __name__ == "__main__":
    main()
