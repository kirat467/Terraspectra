"""First training run: the 3D-CNN on the simulated scene patches, compared with the band-index baselines.

Input : data/generated/patches.npz  (data/make_patches.py: 9 x 9 x 20 patches, block-based train/val/test split)
Steps : standardise each PCA component with TRAIN statistics -> oversample the rare classes in train only
        -> train Crop3DCNN with train_3dcnn.train_model (early stopping, scheduler) -> evaluate once on the test blocks
Output: ml/models/cnn_run/{model.pth, normalization.npz}  (git-ignored), docs/figures/cnn_loss_curve.png, docs/cnn_run_results.json

Classes: 0 healthy, 1 early stress, 2 visible stress. The test blocks are never used to pick settings.
Run from the repository root:   python ml/src/train_on_scenes.py
"""
import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))
sys.path.insert(0, str(ROOT / "data"))

CLASS_NAMES = ["healthy", "early_stress", "visible_stress"]


def load_patches(path):
    z = np.load(path)
    X = z["X"].astype(np.float32)                                  # (N, 9, 9, 20)
    return {"X": X, "y": z["y"].astype(np.int64), "split": z["split"], "pos": z["pos"], "date_index": z["date_index"], "dates": [str(d) for d in z["dates"]]}


def standardise(train_X, *others):
    """Per-component mean and standard deviation from the training patches only."""
    mean = train_X.mean(axis=(0, 1, 2), keepdims=True)
    std = train_X.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    return (mean, std), [(a - mean) / std for a in (train_X, *others)]


def oversample(X, y, target_ratio=0.35, seed=0):
    """Repeat rare-class patches (train only) until each reaches target_ratio of the largest class."""
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=3)
    goal = int(counts.max() * target_ratio)
    idx = [np.arange(len(y))]
    for c in range(3):
        have = np.flatnonzero(y == c)
        if 0 < len(have) < goal:
            idx.append(rng.choice(have, size=goal - len(have), replace=True))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def to_cnn(X):
    return np.ascontiguousarray(X.transpose(0, 3, 1, 2)[:, None])       # (N, 1, comps, h, w)


def parse_history(text):
    train = [float(v) for v in re.findall(r"Training Loss: ([0-9.]+)", text)]
    val = [float(v) for v in re.findall(r"Validation Loss: ([0-9.]+)", text)]
    return train, val


def predict_probs(model, X, batch=256):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.softmax(model(torch.from_numpy(to_cnn(X[i:i + batch]))), dim=1).numpy())
    return np.concatenate(out)


def per_class_report(y, pred):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    p, r, f, n = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {"per_class": {CLASS_NAMES[i]: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3), "f1": round(float(f[i]), 3), "patches": int(n[i])} for i in range(3)},
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(), "accuracy": round(float((pred == y).mean()), 4)}


def baseline_scores(patches, scenes_path, train_mask, test_mask):
    """Scores for the same test patches from NDVI, red edge, water index and a full-spectrum ridge model (centre pixel)."""
    import h5py
    from spectral_baseline import SpectralRidge, disease_score

    pos, di = patches["pos"], patches["date_index"]
    spectra = np.zeros((len(pos), 200), dtype=np.float32)
    severity = np.zeros(len(pos), dtype=np.float32)
    with h5py.File(scenes_path) as h:
        for i, d in enumerate(patches["dates"]):
            m = di == i
            cube = h[f"cube/{d}"][...]
            sev = h[f"severity/{d}"][...]
            spectra[m] = cube[pos[m, 0], pos[m, 1]].astype(np.float32) / float(h.attrs["scale_factor"])
            severity[m] = sev[pos[m, 0], pos[m, 1]]
    scores = {n: disease_score(n, spectra) for n in ("ndvi", "red_edge", "ndii")}
    ridge = SpectralRidge().fit(spectra[train_mask], severity[train_mask])
    scores["full_spectrum_ridge"] = ridge.predict(spectra)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--patches", default=str(ROOT / "data" / "generated" / "patches.npz"))
    ap.add_argument("--scenes", default=str(ROOT / "data" / "generated" / "scenes.h5"))
    ap.add_argument("--out", default=str(ROOT / "ml" / "models" / "cnn_run"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.003)
    a = ap.parse_args()

    import torch
    from torch.utils.data import TensorDataset

    from model_3dcnn import Crop3DCNN
    from spectral_baseline import auc
    from train_3dcnn import save_model, train_model

    P = load_patches(a.patches)
    tr, va, te = (P["split"] == k for k in range(3))
    (mean, std), (Xtr, Xva, Xte) = standardise(P["X"][tr], P["X"][va], P["X"][te])
    ytr, yva, yte = P["y"][tr], P["y"][va], P["y"][te]
    Xtr_b, ytr_b = oversample(Xtr, ytr)
    print("train patches", np.bincount(ytr, minlength=3).tolist(), "-> after oversampling", np.bincount(ytr_b, minlength=3).tolist())

    mk = lambda X, y: TensorDataset(torch.from_numpy(to_cnn(X)), torch.from_numpy(y))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = Crop3DCNN(num_classes=3)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        model = train_model(model, mk(Xtr_b, ytr_b), epochs=a.epochs, batch_size=a.batch, lr=a.lr, validation_dataset=mk(Xva, yva), patience=6,
                            checkpoint_path=str(out_dir / "checkpoint.pth"))
    train_loss, val_loss = parse_history(log.getvalue())
    print(f"trained {len(train_loss)} epochs; best validation loss {min(val_loss):.4f}")
    save_model(model, str(out_dir / "model.pth"))
    np.savez(out_dir / "normalization.npz", mean=mean, std=std)

    probs = predict_probs(model, Xte)
    pred = probs.argmax(axis=1)
    results = {"test_blocks": per_class_report(yte, pred), "epochs_run": len(train_loss), "train_counts": np.bincount(ytr, minlength=3).tolist(),
               "test_counts": np.bincount(yte, minlength=3).tolist(), "val_counts": np.bincount(yva, minlength=3).tolist()}

    # same test patches, every detector: how well does the score separate stressed pixels from healthy ones?
    cnn_score = 1 - probs[:, 0]
    scores = {"cnn_3d": cnn_score, **{k: v[te] for k, v in baseline_scores(P, a.scenes, tr, te).items()}}
    auc_table = {}
    for name, s in scores.items():
        row = {}
        for label, cls in (("early_vs_healthy", 1), ("visible_vs_healthy", 2)):
            pos, neg = s[yte == cls], s[yte == 0]
            row[label] = round(auc(pos, neg), 3) if len(pos) and len(neg) else None
        auc_table[name] = row
    results["auc_on_test_blocks"] = auc_table
    Path(ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "cnn_run_results.json").write_text(json.dumps(results, indent=2))
    plot_loss(train_loss, val_loss, ROOT / "docs" / "figures" / "cnn_loss_curve.png")
    print(json.dumps(results, indent=2))


def plot_loss(train, val, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = np.arange(1, len(train) + 1)
    ax.plot(x, train, color="#2a78d6", linewidth=2.4)
    ax.plot(x[:len(val)], val, color="#eb6834", linewidth=2.4)
    ax.text(x[-1] + 0.3, train[-1] - 0.008, "Training", color="#0b0b0b", va="center", fontsize=11)
    ax.text(x[-1] + 0.3, val[-1] + 0.008, "Validation", color="#0b0b0b", va="center", fontsize=11)
    ax.set_xlim(1, x[-1] + 7)
    ax.set_xlabel("Epoch", color="#52514e"); ax.set_ylabel("Loss", color="#52514e")
    ax.set_title("3D-CNN training on simulated patches", loc="left", fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9"); ax.set_axisbelow(True)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
