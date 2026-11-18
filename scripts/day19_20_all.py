import os, sys, json, glob, ast, hashlib, shutil, subprocess, zipfile
from datetime import date
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (change here) =====================
RUN_REPRO_CHECK = True          # re-runs day16_17_18_all.py and compares file hashes (takes a minute)
AUTHOR = "Kailash Kumar"
REF_NO = "ITS/DSML/1513"
# ==================================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-pptx", "h5py", "pandas"])

import numpy as np
import pandas as pd
import h5py
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

for d in ("docs", "reports", "outputs", "handover"):
    os.makedirs(d, exist_ok=True)


def jl(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else (default if default is not None else {})


def md_table(df):
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    cols = [str(c) for c in df.columns]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(v) for v in r.values) + " |\n"
    return out


def mark(days, hours="3"):
    if os.path.exists("scripts/mark_done.py"):
        for d in days:
            subprocess.run([sys.executable, "scripts/mark_done.py", str(d), hours], stdout=subprocess.DEVNULL)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


H5 = "data/processed/terraspectra_dataset.h5"
if not os.path.exists(H5):
    raise SystemExit("Final dataset not found. Run day16_17_18_all.py first.")
# ---------- gather everything produced so far ----------
dsm = jl("data/metadata/dataset_metadata.json")
quality = jl("data/metadata/quality_report.json")
scene = jl("data/metadata/scene_metadata.json")
smooth = jl("data/metadata/smoothing_choice.json")
cw = jl("data/processed/class_weights.json")
dataset_name = scene.get("dataset", "unknown")
is_test = "Indian" in dataset_name
dist = pd.read_csv("data/metadata/class_distribution_patches.csv") if os.path.exists("data/metadata/class_distribution_patches.csv") else None
fisher = pd.read_csv("data/metadata/indices_fisher_scores.csv") if os.path.exists("data/metadata/indices_fisher_scores.csv") else None
smooth_cmp = pd.read_csv("data/metadata/smoothing_comparison.csv") if os.path.exists("data/metadata/smoothing_comparison.csv") else None
weather_ok = os.path.exists("data/metadata/weather_daily.csv")
n_p = dsm.get("n_patches", {})
patch = dsm.get("patch_size")
bands = dsm.get("bands")
idx_names = dsm.get("indices", [])
classes = dsm.get("classes", {})
# ============================================================
# DAY 19.1: Data dictionary (auto from the real HDF5 file)
# ============================================================
print("\n===== DAY 19: Documentation =====")
rows = []
with h5py.File(H5, "r") as f:
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            rows.append({"path": name, "shape": str(obj.shape), "dtype": str(obj.dtype)})
    f.visititems(visit)
    attrs = {k: (v.decode() if isinstance(v, bytes) else (v.tolist() if hasattr(v, "tolist") else v)) for k, v in f.attrs.items()}
dd = pd.DataFrame(rows)
meaning = {
    "X": "Hyperspectral patches (N, bands, P, P), per-band z-score with TRAIN statistics",
    "X_indices": "Vegetation index patches (N, n_indices, P, P), z-score with TRAIN statistics",
    "y": "Patch label = majority class of labelled pixels",
    "y_pixel": "Pixel-level labels inside each patch (-1 = unlabelled)",
    "valid_mask": "True where the pixel had valid image data (invalid pixels were set to 0 in X)",
    "coords_rc": "Top-left (row, col) of the patch in the scene grid",
    "purity": "Fraction of labelled pixels belonging to the patch label",
    "wavelengths_nm": "Wavelength (nm) of each kept band, same order as X channels",
}
dd["meaning"] = [meaning.get(p.split("/")[-1], "") for p in dd["path"]]
cls_df = pd.DataFrame({"label": list(classes.keys()), "class_name": list(classes.values())})
with open("docs/data_dictionary.md", "w", encoding="utf-8") as f:
    f.write(f"# Data Dictionary - TerraSpectra ({date.today()})\n\n")
    f.write(f"Source data: **{dataset_name}**" + (" (benchmark used to test the pipeline; replace with real scene + disease labels)" if is_test else "") + "\n\n")
    f.write("## File: data/processed/terraspectra_dataset.h5\n\n" + md_table(dd) + "\n")
    f.write("## Global attributes\n\n" + md_table(pd.DataFrame({"key": list(attrs.keys()), "value": [str(v)[:120] for v in attrs.values()]})) + "\n")
    f.write("## Label mapping (new label -> class)\n\n" + md_table(cls_df) + "\n")
    f.write("## Vegetation indices (channel order in X_indices)\n\n" + ", ".join(idx_names) + "\n\n")
    f.write("## Other files\n\n- normalization_train_stats.npz: per-band mean/std (train only) - apply to any new data\n")
    f.write("- split_indices.json: patch indices of train/val/test\n- class_weights.json: balanced class weights + train counts\n")
    f.write("- train_oversampled_idx.npy: oversampled train index list\n- data/metadata/weather_daily.csv: daily weather + disease-risk features\n")
    f.write("\n## Preprocessing order\n\n1. Reflectance scaling/clipping 2. Bad (water) band removal 3. Noisy band removal 4. Cloud/shadow mask 5. Gap filling "
            "6. Reprojection (bilinear; labels nearest) 7. Savitzky-Golay smoothing 8. Vegetation indices 9. Label alignment "
            "10. Patch extraction (non-overlapping) 11. Spatial block split 12. Train-only z-score normalization\n")
print("Saved docs/data_dictionary.md")