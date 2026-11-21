"""Day 23: validate the final dataset with baseline models + leakage demo + week 3 write-up."""
import os, sys, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS =====================
SEED = 42
N_BOOT = 300
TOP_BANDS = 15
# ====================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "h5py", "pandas"])

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

H5 = "data/processed/terraspectra_dataset.h5"
if not os.path.exists(H5):
    raise SystemExit("Final dataset not found. Run day16_17_18_all.py first (needs labels).")
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)

meta = json.load(open("data/metadata/dataset_metadata.json")) if os.path.exists("data/metadata/dataset_metadata.json") else {}
scene = json.load(open("data/metadata/scene_metadata.json")) if os.path.exists("data/metadata/scene_metadata.json") else {}
dataset_name = scene.get("dataset", "unknown")
is_test = "Indian" in dataset_name
class_names = {int(k): v for k, v in meta.get("classes", {}).items()}
# ---------- load + patch-level features ----------
def feats(g):
    X, vm, y = g["X"][:], g["valid_mask"][:], g["y"][:]
    w = vm[:, None].astype("float32")
    f = [(X * w).sum(axis=(2, 3)) / (w.sum(axis=(2, 3)) + 1e-8)]          # mean spectrum over valid pixels
    if "X_indices" in g:
        Xi = g["X_indices"][:]
        f.append((Xi * w).sum(axis=(2, 3)) / (w.sum(axis=(2, 3)) + 1e-8))  # mean vegetation indices
    return np.concatenate(f, axis=1).astype("float32"), y


with h5py.File(H5, "r") as f:
    Xtr, ytr = feats(f["train"]); Xva, yva = feats(f["val"]); Xte, yte = feats(f["test"])
    wl = f["wavelengths_nm"][:]
    n_classes = int(f.attrs["n_classes"])
n_bands = len(wl)
print(f"Features: train {Xtr.shape}, val {Xva.shape}, test {Xte.shape} | classes {n_classes}")
if len(ytr) < 20 or len(np.unique(ytr)) < 2:
    raise SystemExit("Too few train patches/classes for a meaningful baseline. Need more labels.")

cnt = np.bincount(ytr, minlength=n_classes)
print("Train patches per class:", cnt.tolist())


def evaluate(model, X, y):
    p = model.predict(X)
    return {"accuracy": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro", zero_division=0)}, p


models = {
    "majority_class": DummyClassifier(strategy="most_frequent"),
    "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
}

# ============================================================
# 1. Baselines on the spatial split
# ============================================================
print("\n===== Baselines (spatial split) =====")
rows, preds = [], {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    for split, X, y in (("val", Xva, yva), ("test", Xte, yte)):
        if len(y) == 0:
            continue
        r, p = evaluate(m, X, y)
        rows.append({"model": name, "split": split, "n": len(y), **{k: round(v, 4) for k, v in r.items()}})
        preds[(name, split)] = p
res = pd.DataFrame(rows)
res.to_csv("data/metadata/baseline_results.csv", index=False)
print(res.to_string(index=False))