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
# ============================================================
# 2. Bootstrap CI (test accuracy, random forest)
# ============================================================
ci = None
if len(yte) >= 10:
    p = preds[("random_forest", "test")]
    rng = np.random.default_rng(SEED)
    accs = [accuracy_score(yte[i], p[i]) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(N_BOOT))]
    ci = (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))
    print(f"\nRF test accuracy 95% bootstrap CI: {ci[0]:.3f} - {ci[1]:.3f} (n_test={len(yte)})")
# ============================================================
# 3. Leakage demo: random split vs spatial split
# ============================================================
print("\n===== Leakage demo: random vs spatial split =====")
Xall = np.concatenate([Xtr, Xva, Xte]); yall = np.concatenate([ytr, yva, yte])
mc = np.bincount(yall).min() if len(yall) else 0
strat = yall if mc >= 2 else None
Xa, Xb, ya, yb = train_test_split(Xall, yall, test_size=0.3, random_state=SEED, stratify=strat)
rf_rand = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(Xa, ya)
rand_acc = accuracy_score(yb, rf_rand.predict(Xb))
rand_f1 = f1_score(yb, rf_rand.predict(Xb), average="macro", zero_division=0)
sp = res[(res["model"] == "random_forest") & (res["split"] == "test")]
spat_acc = float(sp["accuracy"].iloc[0]) if len(sp) else float("nan")
spat_f1 = float(sp["macro_f1"].iloc[0]) if len(sp) else float("nan")
print(f"Random split : accuracy {rand_acc:.3f} | macro-F1 {rand_f1:.3f}")
print(f"Spatial split: accuracy {spat_acc:.3f} | macro-F1 {spat_f1:.3f}")
gap = rand_acc - spat_acc
print(f"Gap (random - spatial) = {gap:+.3f}  " + ("-> random split is optimistic (neighbouring patches leak information)" if gap > 0.02 else "-> small gap here (patches are non-overlapping; effect is stronger with dense labels)"))
json.dump({"random_split": {"accuracy": rand_acc, "macro_f1": rand_f1},
           "spatial_split": {"accuracy": spat_acc, "macro_f1": spat_f1}, "gap": gap},
          open("data/metadata/leakage_demo.json", "w"), indent=2)

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["random split", "spatial split"], [rand_acc, spat_acc], color=["tab:red", "tab:green"])
ax.set_ylim(0, 1); ax.set_ylabel("accuracy (random forest)"); ax.set_title("Why the spatial split matters")
for i, v in enumerate([rand_acc, spat_acc]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
plt.tight_layout(); plt.savefig("reports/figures/d23_leakage_demo.png", dpi=150); plt.close()
# ============================================================
# 4. Band importance + confusion matrix
# ============================================================
rf = models["random_forest"]
imp = rf.feature_importances_
bimp = imp[:n_bands]
top = np.argsort(bimp)[::-1][:TOP_BANDS]
print(f"\nTop {TOP_BANDS} important wavelengths (nm):", sorted(wl[top].round(0).astype(int).tolist()))
if len(imp) > n_bands:
    inames = meta.get("indices", [f"idx{i}" for i in range(len(imp) - n_bands)])
    print("Index importances:", {n: round(float(v), 4) for n, v in zip(inames, imp[n_bands:])})
pd.DataFrame({"wavelength_nm": wl[top], "importance": bimp[top]}).to_csv("data/metadata/top_bands.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wl, bimp); ax.scatter(wl[top], bimp[top], c="red", s=18, zorder=3)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("RF importance"); ax.set_title("Which spectral regions matter (red = top bands)")
plt.tight_layout(); plt.savefig("reports/figures/d23_band_importance.png", dpi=150); plt.close()

if len(yte):
    cm = confusion_matrix(yte, preds[("random_forest", "test")], labels=list(range(n_classes))).astype(float)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); plt.colorbar(im, label="row-normalized")
    ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Random forest: confusion matrix (test)")
    plt.tight_layout(); plt.savefig("reports/figures/d23_confusion_matrix.png", dpi=150); plt.close()