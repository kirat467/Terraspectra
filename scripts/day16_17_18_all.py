import os, sys, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

# ===================== SETTINGS (change here) =====================
PATCH_SIZE = None            # None = auto (32 / 16 / 8 depending on image size)
MAX_NAN_FRACTION = 0.2       # drop patch if more than this fraction of pixels is invalid
MIN_LABELED_FRACTION = 0.25  # patch needs at least this fraction of labelled pixels
MIN_PURITY = 0.0             # 0 = keep all; e.g. 0.6 keeps only patches dominated by one class
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
N_SPLIT_TRIES = 300          # random block assignments to search
MAX_OVERSAMPLE_FACTOR = 5
# ==================================================================

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for d in ("data/processed", "reports/figures", "data/metadata"):
    os.makedirs(d, exist_ok=True)


def mark(days, hours="3"):
    if os.path.exists("scripts/mark_done.py"):
        for d in days:
            subprocess.run([sys.executable, "scripts/mark_done.py", str(d), hours], stdout=subprocess.DEVNULL)


def git(msg):
    try:
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", msg], check=False)
        if subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip():
            subprocess.run(["git", "push"], check=False)
    except Exception as e:
        print("Git issue:", e)


# ============================================================
# Load inputs (Day 12-15 outputs)
# ============================================================
need = ["data/interim/smoothed_reflectance.npy", "data/interim/labels_aligned.npy",
        "data/interim/wavelengths_clean.npy"]
for p in need:
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found. Run the earlier day scripts first.")

refl = np.load(need[0]).astype("float32")           # (B,H,W) reflectance (smoothed, NOT normalized)
labels = np.load(need[1]).astype("int32")           # (H,W), -1 = unlabelled
wl = np.load(need[2])
B, H, W = refl.shape

if os.path.exists("data/interim/indices_cube.npy"):
    idx_cube = np.load("data/interim/indices_cube.npy").astype("float32")
    idx_names = json.load(open("data/metadata/indices_names.json"))
else:
    idx_cube, idx_names = np.zeros((0, H, W), "float32"), []
K = idx_cube.shape[0]

class_names = {}
if os.path.exists("data/metadata/label_stats.csv"):
    ls = pd.read_csv("data/metadata/label_stats.csv")
    class_names = {int(r.new_label): str(r.class_name) for r in ls.itertuples()}

print(f"Reflectance {refl.shape} | indices {idx_cube.shape} | labels {labels.shape}")
n_classes_total = int(labels.max()) + 1 if labels.max() >= 0 else 0
if n_classes_total == 0:
    raise SystemExit("No labels found (labels_aligned.npy is all -1). Provide labels first (Day 15).")
assert labels.shape == (H, W), "label / image shape mismatch"

if PATCH_SIZE is None:
    m = min(H, W)
    PATCH_SIZE = 32 if m >= 256 else (16 if m >= 96 else 8)
P = PATCH_SIZE
STRIDE = P   # non-overlapping patches => no pixel shared between splits
print(f"Patch size {P}, stride {STRIDE} (non-overlapping)")

nan_px = np.isnan(refl).any(axis=0)

# ============================================================
# DAY 16: Patch extraction
# ============================================================
print("\n===== DAY 16: Patch extraction =====")
coords, y, purity, lab_frac = [], [], [], []
skipped = {"nan": 0, "unlabelled": 0, "impure": 0}
for i in range(0, H - P + 1, STRIDE):
    for j in range(0, W - P + 1, STRIDE):
        if nan_px[i:i + P, j:j + P].mean() > MAX_NAN_FRACTION:
            skipped["nan"] += 1
            continue
        lab = labels[i:i + P, j:j + P]
        lm = lab >= 0
        if lm.mean() < MIN_LABELED_FRACTION:
            skipped["unlabelled"] += 1
            continue
        cnt = np.bincount(lab[lm], minlength=n_classes_total)
        maj = int(cnt.argmax())
        pur = cnt[maj] / lm.sum()
        if pur < MIN_PURITY:
            skipped["impure"] += 1
            continue
        coords.append((i, j)); y.append(maj); purity.append(pur); lab_frac.append(lm.mean())

coords = np.array(coords, dtype="int32")
y = np.array(y, dtype="int32")
purity = np.array(purity, dtype="float32")
N = len(y)
print(f"Patches kept: {N} | skipped: {skipped}")
if N < 20:
    print("WARNING: very few patches. Lower PATCH_SIZE (e.g. 8) or MIN_LABELED_FRACTION.")
print(f"Mean purity: {purity.mean():.2f} | mean labelled fraction: {np.mean(lab_frac):.2f}")


def cut(arr, c):
    return np.stack([arr[:, i:i + P, j:j + P] for i, j in c]).astype("float32")


X_raw = cut(refl, coords)                                   # (N,B,P,P) with NaN where invalid
X_idx_raw = cut(idx_cube, coords) if K else np.zeros((N, 0, P, P), "float32")
Y_map = np.stack([labels[i:i + P, j:j + P] for i, j in coords]).astype("int16")   # (N,P,P) pixel labels
valid_mask = ~np.isnan(X_raw).any(axis=1)                   # (N,P,P)
print("X:", X_raw.shape, "| indices:", X_idx_raw.shape, "| pixel label maps:", Y_map.shape)

# sample figure
gi = [int(np.argmin(np.abs(wl - w))) for w in (800, 650, 550)]
sel = np.random.default_rng(SEED).choice(N, size=min(10, N), replace=False)
fig, ax = plt.subplots(2, 5, figsize=(14, 6))
for a in ax.ravel():
    a.axis("off")
for a, k in zip(ax.ravel(), sel):
    rgb = np.nan_to_num(X_raw[k][gi]).transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, (2, 98))
    a.imshow(np.clip((rgb - lo) / (hi - lo + 1e-9), 0, 1))
    a.set_title(f"y={y[k]} {class_names.get(int(y[k]), '')[:14]}\npurity {purity[k]:.2f}", fontsize=8)
plt.tight_layout(); plt.savefig("reports/figures/d16_sample_patches.png", dpi=140); plt.close()
mark([16])

# ============================================================
# DAY 17: Spatial block split (no random pixel split)
# ============================================================
print("\n===== DAY 17: Spatial split =====")
BS = max(2, min(8, int(min(H, W) / P / 4)))        # block size in patches
block_px = P * BS
block_key = [(int(r) // block_px, int(c) // block_px) for r, c in coords]
uniq_blocks = sorted(set(block_key))
blocks_arr = np.array([uniq_blocks.index(b) for b in block_key])
nb = len(uniq_blocks)
print(f"Block size: {BS}x{BS} patches ({block_px}px) | blocks with data: {nb}")
class_total = np.bincount(y, minlength=n_classes_total)
present = np.where(class_total >= 3)[0]
targets = np.array([RATIOS["train"], RATIOS["val"], RATIOS["test"]])
names3 = ["train", "val", "test"]


def assign(seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(nb)
    counts_per_block = np.bincount(blocks_arr, minlength=nb)
    got = np.zeros(3)
    block_split = np.zeros(nb, dtype=int)
    for b in order:
        deficit = targets * counts_per_block.sum() - got
        s = int(np.argmax(deficit))
        block_split[b] = s
        got[s] += counts_per_block[b]
    return block_split


def score(block_split):
    sp = block_split[blocks_arr]
    frac = np.array([(sp == s).mean() for s in range(3)])
    miss = [0, 0, 0]
    for s in range(3):
        cs = np.bincount(y[sp == s], minlength=n_classes_total)
        miss[s] = int((cs[present] == 0).sum())
    return 100 * miss[0] + 10 * (miss[1] + miss[2]) + 5 * np.abs(frac - targets).sum(), miss, frac


best = None
for t in range(N_SPLIT_TRIES):
    bs_ = assign(SEED + t)
    sc, miss, frac = score(bs_)
    if best is None or sc < best[0]:
        best = (sc, bs_, miss, frac, SEED + t)
_, best_blocks, miss, frac, used_seed = best
split_arr = best_blocks[blocks_arr]
idx = {n: np.where(split_arr == s)[0] for s, n in enumerate(names3)}
print(f"Best split seed: {used_seed} | patch fractions train/val/test: {np.round(frac, 3).tolist()}")
print(f"Classes missing (train, val, test): {miss}   (0,0,0 is ideal)")
if miss[0] > 0:
    print("WARNING: some classes have no train patches. Consider merging rare classes or a smaller patch size.")

# leakage checks
owner = np.zeros((H, W), dtype="int16")
for (i, j) in coords:
    owner[i:i + P, j:j + P] += 1
assert owner.max() <= 1, "Patches overlap in pixels!"
for a_, b_ in (("train", "val"), ("train", "test"), ("val", "test")):
    assert not (set(idx[a_]) & set(idx[b_])), f"{a_}/{b_} share patches"
ctr = coords // P                                      # patch-grid coordinates
if len(idx["train"]) and len(idx["test"]):
    dmin = [np.max(np.abs(ctr[idx["train"]] - c), axis=1).min() for c in ctr[idx["test"]]]
    print(f"Min patch-distance train<->test: min {int(np.min(dmin))}, median {float(np.median(dmin)):.1f} (>=1 means no shared pixels)")

# split map figure
smap = np.zeros((H, W), dtype="int8")
for k, (i, j) in enumerate(coords):
    smap[i:i + P, j:j + P] = split_arr[k] + 1
cmap = matplotlib.colors.ListedColormap(["white", "tab:blue", "tab:orange", "tab:green"])
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(np.ma.masked_equal(labels, -1), cmap="nipy_spectral"); ax[0].set_title("Labels")
ax[1].imshow(smap, cmap=cmap, vmin=0, vmax=3); ax[1].set_title("Spatial split: blue=train orange=val green=test")
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.savefig("reports/figures/d17_spatial_split.png", dpi=150); plt.close()

json.dump({"seed": used_seed, "block_size_patches": BS, "patch_size": P,
           **{k: v.tolist() for k, v in idx.items()}},
          open("data/processed/split_indices.json", "w"))
mark([17])

# ============================================================
# DAY 18: Normalization (train-only), imbalance, final save, validation
# ============================================================
print("\n===== DAY 18: Normalization + imbalance + final dataset =====")
tr = idx["train"]
if len(tr) == 0:
    raise SystemExit("Empty train split. Adjust settings and re-run.")


def train_stats(Xr):
    if Xr.shape[1] == 0:
        return np.zeros(0, "float32"), np.ones(0, "float32")
    t = Xr[tr].transpose(1, 0, 2, 3).reshape(Xr.shape[1], -1)
    return np.nanmean(t, axis=1).astype("float32"), (np.nanstd(t, axis=1) + 1e-8).astype("float32")


mu, sd = train_stats(X_raw)
mu_i, sd_i = train_stats(X_idx_raw)


def norm(Xr, m, s):
    if Xr.shape[1] == 0:
        return Xr
    Z = (Xr - m[None, :, None, None]) / s[None, :, None, None]
    return Z.astype("float32")


X = norm(X_raw, mu, sd)
X_idx = norm(X_idx_raw, mu_i, sd_i)
inv = ~valid_mask[:, None, :, :]
X = np.where(inv, 0.0, X).astype("float32")            # invalid pixels -> 0 (= train mean); use valid_mask to ignore
if K:
    X_idx = np.where(np.isnan(X_idx), 0.0, X_idx).astype("float32")
np.savez("data/processed/normalization_train_stats.npz", mean=mu, std=sd, idx_mean=mu_i, idx_std=sd_i, wavelengths=wl)
print("Normalization stats from TRAIN patches only | train-set mean check: %.4f (should be ~0)" % float(np.mean(X[tr][np.broadcast_to(valid_mask[tr][:, None], X[tr].shape)])))

# class imbalance
rows = []
for n_ in names3:
    cs = np.bincount(y[idx[n_]], minlength=n_classes_total)
    for c in range(n_classes_total):
        rows.append({"split": n_, "class": c, "name": class_names.get(c, f"class_{c}"), "patches": int(cs[c])})
dist = pd.DataFrame(rows)
pivot = dist.pivot(index="class", columns="split", values="patches")[names3]
pivot.insert(0, "name", [class_names.get(c, f"class_{c}") for c in pivot.index])
pivot["total"] = pivot[names3].sum(axis=1)
pivot.to_csv("data/metadata/class_distribution_patches.csv")
print(pivot.to_string())

ctr_train = np.bincount(y[tr], minlength=n_classes_total).astype(float)
present_tr = ctr_train > 0
cw = np.zeros(n_classes_total)
cw[present_tr] = len(tr) / (present_tr.sum() * ctr_train[present_tr])
ratio = ctr_train[present_tr].max() / ctr_train[present_tr].min()
print(f"Train imbalance ratio (max/min): {ratio:.1f}x")

rng = np.random.default_rng(SEED)
target = max(int(np.median(ctr_train[present_tr])), 1)
over_idx = list(tr)
for c in np.where(present_tr)[0]:
    have = int(ctr_train[c])
    if have < target:
        extra = min(target - have, have * (MAX_OVERSAMPLE_FACTOR - 1))
        pool = tr[y[tr] == c]
        over_idx += rng.choice(pool, size=extra, replace=True).tolist()
over_idx = np.array(over_idx, dtype="int64")
np.save("data/processed/train_oversampled_idx.npy", over_idx)
print(f"Oversampling index: {len(tr)} -> {len(over_idx)} train samples (target per class {target}, only indices, data not duplicated)")
rare = [int(c) for c in np.where(present_tr & (ctr_train < 5))[0]]
if rare:
    print("Very rare classes in train (<5 patches):", rare, "-> consider merging/dropping or heavy augmentation")
json.dump({"class_weights_balanced": {int(c): float(cw[c]) for c in range(n_classes_total)},
           "train_patch_counts": {int(c): int(ctr_train[c]) for c in range(n_classes_total)},
           "imbalance_ratio_train": float(ratio), "rare_classes_lt5_train": rare},
          open("data/processed/class_weights.json", "w"), indent=2)

# class distribution figure
fig, ax = plt.subplots(figsize=(11, 4.5))
wbar = 0.27
xs = np.arange(n_classes_total)
for k, n_ in enumerate(names3):
    ax.bar(xs + (k - 1) * wbar, pivot[n_].values, wbar, label=n_)
ax.set_xlabel("class"); ax.set_ylabel("patches"); ax.set_title("Patch count per class per split"); ax.legend()
plt.tight_layout(); plt.savefig("reports/figures/d18_class_distribution_splits.png", dpi=150); plt.close()

# ---------- save final dataset ----------
out = "data/processed/terraspectra_dataset.h5"
with h5py.File(out, "w") as f:
    for n_ in names3:
        g = f.create_group(n_)
        ii = idx[n_]
        g.create_dataset("X", data=X[ii], compression="gzip", compression_opts=4)
        if K:
            g.create_dataset("X_indices", data=X_idx[ii], compression="gzip", compression_opts=4)
        g.create_dataset("y", data=y[ii])
        g.create_dataset("y_pixel", data=Y_map[ii], compression="gzip")
        g.create_dataset("valid_mask", data=valid_mask[ii])
        g.create_dataset("coords_rc", data=coords[ii])
        g.create_dataset("purity", data=purity[ii])
    f.create_dataset("wavelengths_nm", data=wl)
    f.attrs.update({"patch_size": P, "n_bands": B, "n_indices": K, "indices": json.dumps(idx_names),
                    "n_classes": n_classes_total, "split_seed": used_seed, "created": str(date.today()),
                    "normalization": "per-band z-score using TRAIN patch statistics only",
                    "label_unlabelled_value": -1, "layout": "X: (N, bands, P, P)"})
print("Saved", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")

np.save("data/processed/X.npy", X); np.save("data/processed/y.npy", y)
np.save("data/processed/coords.npy", coords)
if K:
    np.save("data/processed/X_indices.npy", X_idx)
json.dump({"patch_size": P, "bands": B, "wavelengths_nm": wl.tolist(), "indices": idx_names,
           "classes": {int(c): class_names.get(c, f"class_{c}") for c in range(n_classes_total)},
           "n_patches": {k: int(len(v)) for k, v in idx.items()}, "split_seed": used_seed,
           "block_size_patches": BS},
          open("data/metadata/dataset_metadata.json", "w"), indent=2)

# ---------- validation (reload from disk) ----------
print("\n--- Validation ---")
with h5py.File(out, "r") as f:
    for n_ in names3:
        Xs, ys = f[n_]["X"], f[n_]["y"][:]
        assert Xs.shape[0] == len(ys) == len(idx[n_]), f"{n_}: size mismatch"
        assert not np.isnan(Xs[:]).any() and not np.isinf(Xs[:]).any(), f"{n_}: NaN/inf in X"
        assert ys.min() >= 0 and ys.max() < n_classes_total, f"{n_}: label out of range"
        print(f"OK {n_:5s} X{Xs.shape} classes={sorted(set(ys.tolist()))}")
assert not (set(idx["train"]) & set(idx["val"]) | set(idx["train"]) & set(idx["test"]) | set(idx["val"]) & set(idx["test"]))
print("OK: no NaN/inf, labels in range, no split overlap, no shared pixels, train-only normalization")

with open("docs/dataset_card.md", "w", encoding="utf-8") as f:
    f.write(f"# TerraSpectra Dataset Card ({date.today()})\n\n")
    f.write(f"- File: data/processed/terraspectra_dataset.h5 (groups: train / val / test)\n")
    f.write(f"- X: (N, {B}, {P}, {P}) float32, z-score per band (train statistics)\n")
    f.write(f"- X_indices: (N, {K}, {P}, {P}) {idx_names}\n- y: patch label (majority); y_pixel: pixel labels (-1 unlabelled); valid_mask: usable pixels\n")
    f.write(f"- Patches: {N} -> train {len(idx['train'])}, val {len(idx['val'])}, test {len(idx['test'])}\n")
    f.write(f"- Classes: {n_classes_total} | train imbalance ratio {ratio:.1f}x | class weights: class_weights.json\n")
    f.write(f"- Split: spatial blocks {BS}x{BS} patches, seed {used_seed}; no pixel shared across splits\n")
    f.write("- Normalization stats: normalization_train_stats.npz (apply the same to any new data)\n")
    f.write("- Oversampling: train_oversampled_idx.npy (indices only)\n")

mark([18])
git("Day 16-18: patches, spatial split, imbalance handling, final dataset")
print("\nDAY 16-18 DONE. Figures: d16_sample_patches, d17_spatial_split, d18_class_distribution_splits")
print("Final dataset: data/processed/terraspectra_dataset.h5 | card: docs/dataset_card.md")