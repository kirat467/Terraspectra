import os, sys, glob, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
cfg = yaml.safe_load(open("config/config.yaml"))


def git(msg, tag=None):
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", msg], check=False)
        if tag:
            subprocess.run(["git", "tag", "-f", tag], check=True)
        if subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip():
            subprocess.run(["git", "push"], check=False)
            if tag:
                subprocess.run(["git", "push", "-f", "origin", tag], check=False)
    except Exception as e:
        print("Git issue:", e)


def mark(days, hours="3"):
    if os.path.exists("scripts/mark_done.py"):
        for d in days:
            subprocess.run([sys.executable, "scripts/mark_done.py", str(d), hours], stdout=subprocess.DEVNULL)


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}
# ============================================================
# DAY 10: Phase 1 review + mentor update
# ============================================================
print("\n===== DAY 10: Phase 1 review =====")
scene = load_json("data/metadata/scene_metadata.json")
quality = load_json("data/metadata/quality_report.json")
dataset_name = scene.get("dataset", "unknown")
is_test_data = "Indian" in dataset_name

# tracker status
tracker = pd.read_csv("progress_tracker.csv", dtype=str).fillna("")
done = tracker[(tracker["Day"].astype(int) <= 9) & (tracker["Status"] == "Done")]
print(f"Days 1-9 done in tracker: {len(done)}/9")

# auto-detected pending issues
issues = []
if is_test_data:
    issues.append("No real hyperspectral scene + disease labels yet (using Indian Pines only for pipeline testing).")
    issues.append("Georeference is a placeholder for test data; real scene CRS/transform needed.")
if not glob.glob("data/raw/emit/*.nc"):
    issues.append("No EMIT/EnMAP/PRISMA scene downloaded yet.")
if quality.get("nan_percent_final", 0) > 5:
    issues.append(f"Final NaN % is high ({quality['nan_percent_final']}%): check masks.")
issues.append("Disease labels source still to be confirmed by mentor." if is_test_data else "Verify label quality with mentor.")
issues.append("Cloud mask is heuristic; official EMIT MASK product recommended for real data.")

# figure contact sheet
figs = sorted(glob.glob("reports/figures/*.png"))
key = [f for f in figs if os.path.basename(f).startswith(("eda_mean", "eda_composites", "d7_bad", "d8_spectra", "d9_align"))][:6]
if key:
    n = len(key)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    ax = np.atleast_1d(ax).ravel()
    for a in ax:
        a.axis("off")
    for a, f in zip(ax, key):
        a.imshow(mpimg.imread(f)); a.set_title(os.path.basename(f), fontsize=8)
    plt.tight_layout(); plt.savefig("reports/figures/phase1_contact_sheet.png", dpi=130); plt.close()

# Phase 1 summary
with open("docs/phase1_summary.md", "w", encoding="utf-8") as f:
    f.write(f"# Phase 1 Summary (Day 1-10) - {date.today()}\n\n")
    f.write("## Completed\n- Environment, git repo, mentor questions\n- Data sources registration and search\n")
    f.write("- Label dataset shortlist, Indian Pines pipeline test\n- Raw inventory, metadata inspection, EDA\n")
    f.write("- Bad band removal, noisy band detection, masking, gap filling\n- Reprojection/resampling and label alignment check\n\n")
    f.write(f"## Data used\n- {dataset_name}\n- Shape (B,H,W): {scene.get('shape_BHW')}\n\n")
    f.write("## Cleaning results\n")
    for k, v in quality.items():
        f.write(f"- {k}: {v}\n")
    f.write("\n## Pending issues\n")
    for i in issues:
        f.write(f"- {i}\n")
    f.write("\n## Phase 2 plan\n- Normalization, smoothing, vegetation indices, weather data\n")
    f.write("- Label alignment, patches, spatial split, final dataset, documentation, handover\n")
print("Saved docs/phase1_summary.md")

# mentor update text
msg = f"""Subject: Phase 1 Progress Update - TerraSpectra (Ref: ITS/DSML/1513)

Hello Sir/Ma'am,

Phase 1 (Data Collection & Cleaning) is complete. Summary:

- Environment, git repo and project structure ready
- Hyperspectral sources reviewed (EMIT, EnMAP, PRISMA, DESIS) and accounts set up
- Pipeline built and tested on {dataset_name}
- Bands: {quality.get('bands_original')} -> {quality.get('bands_after_cleaning')} after water-absorption and noisy band removal
- Final NaN percentage: {quality.get('nan_percent_final')}%
- Reprojection/resampling and label alignment check done
- EDA figures and quality report available in the repo

Blockers / need your input:
""" + "\n".join(f"{i+1}. {x}" for i, x in enumerate(issues)) + """

Next (Phase 2): normalization, smoothing, vegetation indices, label alignment, patch creation, spatial train/val/test split and final dataset handover.

Regards,
Kailash Kumar
"""
open("docs/mentor_update_phase1.txt", "w", encoding="utf-8").write(msg)
print("Saved docs/mentor_update_phase1.txt (send this to your mentor)")

mark([10])
git("Day 10: Phase 1 summary and mentor update", tag="phase1-complete")
print("Git tag phase1-complete created")

# ============================================================
# Load working cube (reprojected if available)
# ============================================================
if os.path.exists("data/interim/cleaned_reprojected.tif"):
    import rasterio
    with rasterio.open("data/interim/cleaned_reprojected.tif") as s:
        cube = s.read().astype("float32")
    gt_path = "data/interim/gt_reprojected.npy"
    labels = np.load(gt_path) if os.path.exists(gt_path) else None
    src_used = "cleaned_reprojected.tif"
elif os.path.exists("data/interim/cleaned_cube.npy"):
    cube = np.load("data/interim/cleaned_cube.npy").astype("float32")
    labels = np.load("data/interim/indian_pines_gt.npy") if os.path.exists("data/interim/indian_pines_gt.npy") else None
    src_used = "cleaned_cube.npy"
else:
    raise SystemExit("Cleaned data not found. Run day7_8_9_all.py first.")
wl = np.load("data/interim/wavelengths_clean.npy")
B, H, W = cube.shape
print(f"\nWorking cube ({src_used}): {cube.shape}")

valid_px = ~np.isnan(cube).any(axis=0)
flat = cube.reshape(B, -1)
rng = np.random.default_rng(42)
vidx = np.where(valid_px.ravel())[0]
samp = flat[:, rng.choice(vidx, size=min(5000, len(vidx)), replace=False)]

# ============================================================
# DAY 11: Normalization comparison
# ============================================================
print("\n===== DAY 11: Normalization =====")
mu = np.nanmean(cube, axis=(1, 2), keepdims=True)
sd = np.nanstd(cube, axis=(1, 2), keepdims=True)
lo = np.nanpercentile(flat, 1, axis=1)[:, None, None]
hi = np.nanpercentile(flat, 99, axis=1)[:, None, None]

z_cube = (cube - mu) / (sd + 1e-8)
m_cube = np.clip((cube - lo) / (hi - lo + 1e-8), 0, 1)

print("Z-score  : mean %.3f std %.3f" % (np.nanmean(z_cube), np.nanstd(z_cube)))
print("Min-max  : min %.3f max %.3f mean %.3f" % (np.nanmin(m_cube), np.nanmax(m_cube), np.nanmean(m_cube)))

bands_show = [int(np.argmin(np.abs(wl - w))) for w in (550, 800, 1600)]
fig, ax = plt.subplots(3, 3, figsize=(12, 8))
for r, b in enumerate(bands_show):
    for c, (name, arr) in enumerate((("raw", cube), ("z-score", z_cube), ("min-max", m_cube))):
        v = arr[b][~np.isnan(arr[b])]
        ax[r, c].hist(v, bins=50)
        ax[r, c].set_title(f"{name} band {b} ({wl[b]:.0f} nm)", fontsize=9)
plt.tight_layout(); plt.savefig("reports/figures/d11_normalization.png", dpi=150); plt.close()

norm_stats = {
    "method_comparison": ["per_band_zscore", "per_band_minmax_p1_p99"],
    "zscore_mean_per_band": mu.ravel().tolist(),
    "zscore_std_per_band": sd.ravel().tolist(),
    "minmax_p1_per_band": lo.ravel().tolist(),
    "minmax_p99_per_band": hi.ravel().tolist(),
    "note": "Scene-level stats for exploration. FINAL normalization will use TRAIN-split stats only (Day 17-18) to avoid leakage.",
}
json.dump(norm_stats, open("data/metadata/normalization_stats.json", "w"))
mark([11])
print("Saved normalization_stats.json | recommendation: z-score for deep models, min-max for tree/linear baselines")

# ============================================================
# DAY 12: Savitzky-Golay smoothing
# ============================================================
print("\n===== DAY 12: Spectral smoothing =====")


def roughness(x):
    return float(np.abs(np.diff(x, n=2, axis=0)).mean())


base_r = roughness(samp)
rows = []
for win in (5, 7, 9, 11):
    for poly in (2, 3):
        if win <= poly:
            continue
        sm = savgol_filter(samp, window_length=win, polyorder=poly, axis=0)
        rows.append({"window": win, "polyorder": poly,
                     "roughness": roughness(sm),
                     "roughness_reduction_pct": 100 * (1 - roughness(sm) / base_r),
                     "rmse_vs_raw": float(np.sqrt(np.mean((sm - samp) ** 2)))})
res = pd.DataFrame(rows).sort_values(["window", "polyorder"])
print("Raw roughness:", round(base_r, 6))
print(res.round(6).to_string(index=False))

cand = res[res["roughness_reduction_pct"] >= 30]
best = (cand.sort_values("rmse_vs_raw").iloc[0] if len(cand)
        else res.sort_values("roughness_reduction_pct", ascending=False).iloc[0])
WIN, POLY = int(best["window"]), int(best["polyorder"])
print(f"\nSelected: window={WIN}, polyorder={POLY} "
      f"(roughness -{best['roughness_reduction_pct']:.1f}%, RMSE {best['rmse_vs_raw']:.5f})")
res.to_csv("data/metadata/smoothing_comparison.csv", index=False)
json.dump({"savgol_window": WIN, "savgol_polyorder": POLY},
          open("data/metadata/smoothing_choice.json", "w"))

# before/after plot
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in range(3):
    ax[k].plot(wl, samp[:, k], label="raw", alpha=0.7)
    ax[k].plot(wl, savgol_filter(samp[:, k], WIN, POLY), label=f"SG({WIN},{POLY})")
    ax[k].set_xlabel("Wavelength (nm)"); ax[k].set_title(f"pixel sample {k}")
ax[0].legend(); plt.tight_layout(); plt.savefig("reports/figures/d12_smoothing.png", dpi=150); plt.close()

# apply to whole cube (NaN pixels stay NaN)
nanmask = np.isnan(cube).any(axis=0)
smooth = savgol_filter(np.nan_to_num(cube, nan=0.0), WIN, POLY, axis=0).astype("float32")
smooth[:, nanmask] = np.nan
np.save("data/interim/smoothed_reflectance.npy", smooth)

# smoothed -> z-score = preprocessed cube (scene stats; final uses train stats later)
mu2 = np.nanmean(smooth, axis=(1, 2), keepdims=True)
sd2 = np.nanstd(smooth, axis=(1, 2), keepdims=True)
pre = ((smooth - mu2) / (sd2 + 1e-8)).astype("float32")
np.save("data/interim/preprocessed_cube.npy", pre)
if labels is not None:
    np.save("data/interim/preprocessed_labels.npy", labels)
print("Saved data/interim/smoothed_reflectance.npy and preprocessed_cube.npy", pre.shape)

# class-wise spectra after smoothing
if labels is not None and labels.shape == (H, W):
    plt.figure(figsize=(10, 5))
    for c in np.unique(labels):
        if c == 0:
            continue
        m = (labels == c) & ~nanmask
        if m.sum() > 0:
            plt.plot(wl, smooth[:, m].mean(axis=1), label=f"class {c}")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Reflectance"); plt.title("Class-wise spectra after smoothing")
    plt.legend(ncol=4, fontsize=7); plt.tight_layout()
    plt.savefig("reports/figures/d12_classwise_smoothed.png", dpi=150); plt.close()

with open("docs/day10_12_summary.md", "w", encoding="utf-8") as f:
    f.write(f"# Day 10-12 Summary ({date.today()})\n\n- Phase 1 summary: docs/phase1_summary.md (tag phase1-complete)\n")
    f.write(f"- Normalization: z-score vs min-max compared; final uses train-split stats\n")
    f.write(f"- Smoothing: Savitzky-Golay window={WIN}, polyorder={POLY} "
            f"(roughness -{best['roughness_reduction_pct']:.1f}%)\n- Output: data/interim/preprocessed_cube.npy {pre.shape}\n")

mark([12])
git("Day 11-12: normalization comparison and Savitzky-Golay smoothing")
print("\nDAY 10-12 DONE.")
print("Send docs/mentor_update_phase1.txt to your mentor. Figures: d11_normalization.png, d12_smoothing.png")