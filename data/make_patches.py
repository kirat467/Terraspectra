"""Cut the scans into small patches for the 3D-CNN, with a split that cannot leak.

Steps
  1. Divide the farm into square blocks (18 x 18 pixels by default) and give each block to train, validation or test.
     Blocks with disease are shared out first so every split sees early and visible stress.
  2. Fit PCA (200 -> 20 bands) on TRAIN pixels only, then apply it everywhere. Fitting on all pixels would let test
     information shape the features.
  3. Cut 9 x 9 x 20 patches whose whole footprint, with a 2-pixel margin, lies inside one block. A patch never shares
     a pixel with a patch from another split, and there are at least 4 pixels between them.
  4. The patch label is the label of its centre pixel (0 healthy, 1 early stress, 2 visible stress). Patches centred on
     tracks, ponds or fallow land are skipped.
The same place is in the same split on every date, so the model is never tested on a place it trained on at another date.

Writes data/generated/patches.npz (git-ignored), docs/figures/pca_variance.png and patches_summary.json.

    python data/make_patches.py
Needs scikit-learn (ml/src/pca.py) and matplotlib.
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

SPLITS = ("train", "val", "test")


def assign_blocks(label_stack, block=18, fractions=(0.6, 0.2, 0.2), seed=42):
    """Split id (0 train, 1 val, 2 test) for each block. label_stack: (dates, H, W) labels. Returns (nby, nbx) array."""
    _, h, w = label_stack.shape
    nby, nbx = -(-h // block), -(-w // block)
    stress = np.zeros((nby, nbx))
    for by in range(nby):
        for bx in range(nbx):
            tile = label_stack[:, by * block:(by + 1) * block, bx * block:(bx + 1) * block]
            stress[by, bx] = (tile >= 1).sum()
    flat_stress = stress.ravel()
    rng = np.random.default_rng(seed)
    out = np.zeros(nby * nbx, dtype=int)
    target = np.array(fractions) * out.size
    counts = np.zeros(3)
    # diseased blocks first, most diseased first, each to whichever split is furthest below its share of blocks
    diseased = sorted(np.flatnonzero(flat_stress > 0), key=lambda i: -flat_stress[i])
    clean = list(rng.permutation(np.flatnonzero(flat_stress == 0)))
    for i in diseased + clean:
        shortfall = target - counts
        k = int(np.argmax(shortfall + 1e-6 * rng.random(3)))
        out[i] = k
        counts[k] += 1
    # the biggest diseased blocks are spread over the splits by the rule above, but the very first one would always
    # land in train; swap so each split gets at least one of the top-3 diseased blocks
    top = diseased[:3]
    if len(top) == 3 and len({out[i] for i in top}) < 3:
        for i, k in zip(top, (0, 1, 2)):
            old = out[i]
            if old != k:
                partner = [j for j in range(out.size) if out[j] == k and j not in top]
                if partner:
                    out[partner[0]] = old
                    out[i] = k
    return out.reshape(nby, nbx)


def cut_patches(pca_cube, label, block_split, block=18, patch=9, stride=1, margin=2, max_nonlabel=0.2):
    """Patches (N, patch, patch, comps), labels, split ids and centre positions for one date."""
    h, w, _ = pca_cube.shape
    half = patch // 2
    xs, ys, sp, pos = [], [], [], []
    for by in range(block_split.shape[0]):
        for bx in range(block_split.shape[1]):
            y0, y1 = by * block + margin, min((by + 1) * block, h) - margin
            x0, x1 = bx * block + margin, min((bx + 1) * block, w) - margin
            for r in range(y0, y1 - patch + 1, stride):
                for c in range(x0, x1 - patch + 1, stride):
                    centre = label[r + half, c + half]
                    if centre < 0 or (label[r:r + patch, c:c + patch] < 0).mean() > max_nonlabel:
                        continue
                    xs.append(pca_cube[r:r + patch, c:c + patch, :])
                    ys.append(centre)
                    sp.append(block_split[by, bx])
                    pos.append((r + half, c + half))
    if not xs:
        return np.zeros((0, patch, patch, pca_cube.shape[2]), np.float32), np.zeros(0, np.int8), np.zeros(0, np.int8), np.zeros((0, 2), int)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int8), np.array(sp, dtype=np.int8), np.array(pos)


def to_cnn_layout(patches):
    """(N, h, w, comps) -> (N, 1, comps, h, w), the input layout of Crop3DCNN."""
    return np.ascontiguousarray(patches.transpose(0, 3, 1, 2)[:, None])


def build(scenes_path, n_components=20, block=18, patch=9, stride=1, dates=None, seed=42):
    from pca import fit_pca, transform_pca

    dates = dates or list_dates(scenes_path)
    scenes = [load_scene(scenes_path, d) for d in dates]
    labels = np.stack([s["label"] for s in scenes])
    blocks = assign_blocks(labels, block, seed=seed)
    # PCA on train-block pixels only
    h, w = labels.shape[1:]
    train_mask = np.kron(blocks == 0, np.ones((block, block), bool))[:h, :w]
    pixels = np.concatenate([s["cube"][train_mask] for s in scenes])
    pca = fit_pca(pixels[:, None, :], n_components)
    X, y, sp, pos, dt = [], [], [], [], []
    for i, s in enumerate(scenes):
        reduced = transform_pca(s["cube"], pca)
        px, py, ps, pp = cut_patches(reduced, s["label"], blocks, block, patch, stride)
        X.append(px); y.append(py); sp.append(ps); pos.append(pp); dt.append(np.full(len(py), i, dtype=np.int8))
    return {"X": np.concatenate(X), "y": np.concatenate(y), "split": np.concatenate(sp), "pos": np.concatenate(pos),
            "date_index": np.concatenate(dt), "dates": dates, "blocks": blocks, "pca": pca}


def summarise(out):
    s = {"dates": out["dates"], "patch_shape": list(out["X"].shape[1:]), "total": int(len(out["y"])),
         "explained_variance_20": round(float(out["pca"].explained_variance_ratio_.sum()), 4), "splits": {}}
    for k, name in enumerate(SPLITS):
        m = out["split"] == k
        s["splits"][name] = {"patches": int(m.sum()), "healthy": int((out["y"][m] == 0).sum()),
                             "early_stress": int((out["y"][m] == 1).sum()), "visible_stress": int((out["y"][m] == 2).sum())}
    return s


def plot_variance(pca, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratio = pca.explained_variance_ratio_
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = np.arange(1, len(ratio) + 1)
    ax.bar(x, ratio * 100, color="#2a78d6", width=0.65)
    ax.set_yscale("log")
    ax.set_ylim(min(ratio.min() * 100 / 3, 1e-3), 200)
    ax.text(0.98, 0.93, f"Components 1 to 3 hold {cum[2] * 100:.1f}% of the variance.\nAll {len(ratio)} hold {cum[-1] * 100:.2f}%.",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color="#0b0b0b")
    ax.set_xticks(x[::2]); ax.set_xlim(0.3, len(ratio) + 0.7)
    ax.set_xlabel("Principal component", color="#52514e"); ax.set_ylabel("Share of spectral variance (%, log scale)", color="#52514e")
    ax.set_title("200 bands reduced to 20 components (fitted on training pixels only)", loc="left", fontsize=12, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9"); ax.set_axisbelow(True)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenes", default=str(ROOT / "data" / "generated" / "scenes.h5"))
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "patches.npz"))
    ap.add_argument("--figure", default=str(ROOT / "docs" / "figures" / "pca_variance.png"))
    a = ap.parse_args()
    out = build(a.scenes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, X=out["X"].astype(np.float16), y=out["y"], split=out["split"], pos=out["pos"], date_index=out["date_index"],
                        dates=np.array(out["dates"]), blocks=out["blocks"], pca_mean=out["pca"].mean_, pca_components=out["pca"].components_)
    plot_variance(out["pca"], a.figure)
    summary = summarise(out)
    (Path(a.out).parent / "patches_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
