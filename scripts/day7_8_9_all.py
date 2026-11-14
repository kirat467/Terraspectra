import os, sys, glob, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

import numpy as np
import pandas as pd
import yaml
import scipy.io as sio
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)

cfg = yaml.safe_load(open("config/config.yaml"))
PP = cfg["preprocessing"]
BAD_RANGES = cfg["sensor"]["bad_band_ranges_nm"]
CROP_SIZE = 512
# ============================================================
# Loader (same as Day 4-6): EMIT .nc or Indian Pines fallback
# ============================================================
def load_data():
    nc_files = glob.glob("data/raw/emit/*.nc")
    if nc_files:
        import xarray as xr
        p = nc_files[0]
        print("Using EMIT scene:", p)
        ds = xr.open_dataset(p)
        refl = ds["reflectance"]
        bp = xr.open_dataset(p, group="sensor_band_parameters")
        wl = bp["wavelengths"].values.astype("float32")
        good = bp["good_wavelengths"].values.astype(bool) if "good_wavelengths" in bp else np.ones(len(wl), bool)
        H, W = refl.shape[0], refl.shape[1]
        r0, c0 = max(0, H // 2 - CROP_SIZE // 2), max(0, W // 2 - CROP_SIZE // 2)
        sub = refl[r0:r0 + CROP_SIZE, c0:c0 + CROP_SIZE, :].values.astype("float32")
        sub[sub <= -9990] = np.nan
        return dict(name=os.path.basename(p), sensor="EMIT", cube=np.transpose(sub, (2, 0, 1)),
                    wl=wl, gt=None, good=good)
    cp, gp = "data/interim/indian_pines_cube.npy", "data/interim/indian_pines_gt.npy"
    if os.path.exists(cp):
        cube, gt = np.load(cp), np.load(gp)
    else:
        mp = "data/raw/indian_pines/Indian_pines_corrected.mat"
        gpm = "data/raw/indian_pines/Indian_pines_gt.mat"
        if not os.path.exists(mp):
            raise SystemExit("Data not found. Run day2_day3_all.py first.")
        cube = np.transpose(sio.loadmat(mp)["indian_pines_corrected"], (2, 0, 1)).astype("float32")
        gt = sio.loadmat(gpm)["indian_pines_gt"].astype("int32")
    wl = np.linspace(400, 2500, cube.shape[0]).astype("float32")  # approximate
    return dict(name="Indian_Pines", sensor="INDIAN_PINES", cube=cube.astype("float32"),
                wl=wl, gt=gt, good=np.ones(cube.shape[0], bool))


D = load_data()
cube, wl, gt, good, sensor = D["cube"], D["wl"], D["gt"], D["good"], D["sensor"]
B0 = cube.shape[0]
print(f"Loaded {D['name']} | shape (B,H,W) = {cube.shape}")
# ============================================================
# DAY 7: bad bands, nodata, cloud/shadow mask
# ============================================================
print("\n===== DAY 7: Bad bands + masking =====")

# 1) reflectance scaling (only if values look like integer DN)
if np.nanmax(cube) > 2:
    cube = cube / PP["reflectance_scale"]
    print("Scaled by", PP["reflectance_scale"])
# 2) bad band removal
if sensor == "INDIAN_PINES":
    # this version already has water bands removed and wavelengths are approximate,
    # so wavelength-range removal is skipped on purpose
    bad_water = np.array([], dtype=int)
    print("Indian Pines: wavelength-range removal skipped (water bands already removed in dataset)")
else:
    in_range = np.zeros(len(wl), bool)
    for lo, hi in BAD_RANGES:
        in_range |= (wl >= lo) & (wl <= hi)
    bad_water = np.where(in_range | ~good)[0]
    print("Water-absorption / flagged bands:", len(bad_water))

keep = np.setdiff1d(np.arange(cube.shape[0]), bad_water)
cube, wl_k = cube[keep], wl[keep]
# 3) noisy band detection: deviation of each band from the average of its neighbours
rng = np.random.default_rng(42)
flat = cube.reshape(cube.shape[0], -1)
valid_px = np.where(~np.isnan(flat).any(axis=0))[0]
pick = rng.choice(valid_px, size=min(5000, len(valid_px)), replace=False)
S = flat[:, pick]
local_noise = np.zeros(S.shape[0])
local_noise[1:-1] = np.abs(S[1:-1] - 0.5 * (S[:-2] + S[2:])).mean(axis=1)
local_noise[0], local_noise[-1] = local_noise[1], local_noise[-2]
z = (local_noise - local_noise.mean()) / (local_noise.std() + 1e-12)
noisy = np.where(z > 3)[0]
print("Noisy bands (z>3):", len(noisy), "->", wl_k[noisy].round(0).tolist())
keep2 = np.setdiff1d(np.arange(cube.shape[0]), noisy)
cube, wl_c = cube[keep2], wl_k[keep2]

# plot removed bands
plt.figure(figsize=(10, 4))
full_mean = np.nanmean(D["cube"].reshape(B0, -1), axis=1)
full_mean = full_mean / (PP["reflectance_scale"] if np.nanmax(D["cube"]) > 2 else 1)
plt.plot(wl, full_mean, color="gray", label="original mean spectrum")
for b in bad_water:
    plt.axvline(wl[b], color="red", alpha=0.15)
plt.plot(wl_c, np.nanmean(cube.reshape(cube.shape[0], -1), axis=1), color="green", label="kept bands")
plt.xlabel("Wavelength (nm)"); plt.ylabel("Mean value"); plt.title("Bad band removal (red = removed)")
plt.legend(); plt.tight_layout(); plt.savefig("reports/figures/d7_bad_band_removal.png", dpi=150); plt.close()
print(f"Bands: {B0} -> {cube.shape[0]}")
# 4) cloud / shadow mask (heuristic, EMIT only)
H, W = cube.shape[1:]
cloud = np.zeros((H, W), bool)
shadow = np.zeros((H, W), bool)
if sensor == "EMIT":
    def sel(lo, hi):
        return (wl_c >= lo) & (wl_c <= hi)
    vis = np.nanmean(cube[sel(450, 690)], axis=0)
    nir = np.nanmean(cube[sel(800, 900)], axis=0)
    cloud = binary_dilation(np.nan_to_num(vis) > 0.30, iterations=2)
    shadow = binary_dilation(np.nan_to_num(nir, nan=1) < 0.03, iterations=1)
    cube[:, cloud | shadow] = np.nan
    print(f"Cloud pixels: {cloud.mean()*100:.2f}% | shadow pixels: {shadow.mean()*100:.2f}% (heuristic mask)")
    print("Tip: for exact masks use the EMIT L2A MASK product and replace this heuristic.")
else:
    print("Indian Pines: no cloud mask needed")
np.save("data/interim/mask_cloud.npy", cloud)
np.save("data/interim/mask_shadow.npy", shadow)
# ============================================================
# DAY 8: gap filling + quality checks
# ============================================================
print("\n===== DAY 8: Gap filling + quality checks =====")
before = cube.copy()


def fill_gaps(c, max_nan_frac=0.2):
    B, H, W = c.shape
    f = c.reshape(B, -1).copy()
    frac = np.isnan(f).mean(axis=0)
    partial = np.where((frac > 0) & (frac <= max_nan_frac))[0]
    x = np.arange(B)
    for i in partial:
        col = f[:, i]
        m = np.isnan(col)
        col[m] = np.interp(x[m], x[~m], col[~m])
    drop = frac > max_nan_frac
    f[:, drop] = np.nan
    return f.reshape(B, H, W), len(partial), int(drop.sum())


cube, n_filled, n_dropped = fill_gaps(cube, cfg["dataset"]["max_nan_fraction"])
cube = np.clip(cube, 0.0, 1.0)
print(f"Pixels interpolated: {n_filled} | pixels dropped (too many NaN): {n_dropped}")


def smoothness(c):
    f = c.reshape(c.shape[0], -1)
    f = f[:, ~np.isnan(f).any(axis=0)][:, :5000]
    return float(np.abs(np.diff(f, n=2, axis=0)).mean())


report = {
    "bands_original": int(B0),
    "bands_after_cleaning": int(cube.shape[0]),
    "water_bands_removed": int(len(bad_water)),
    "noisy_bands_removed": int(len(noisy)),
    "nan_percent_final": round(float(np.isnan(cube).mean() * 100), 3),
    "pixels_interpolated": int(n_filled),
    "pixels_dropped": int(n_dropped),
    "negative_values_before_clip_pct": round(float((before < 0).sum() / before.size * 100), 4),
    "spectral_roughness": round(smoothness(cube), 6),
    "value_min": float(np.nanmin(cube)),
    "value_max": float(np.nanmax(cube)),
}
print(json.dumps(report, indent=2))
# spot-check spectra plot
ok = np.where(~np.isnan(cube).any(axis=0))
sel_idx = rng.choice(len(ok[0]), size=min(8, len(ok[0])), replace=False)
plt.figure(figsize=(10, 4))
for k in sel_idx:
    plt.plot(wl_c, cube[:, ok[0][k], ok[1][k]], alpha=0.7)
plt.xlabel("Wavelength (nm)"); plt.ylabel("Reflectance"); plt.title("Random pixel spectra after cleaning (should look smooth)")
plt.tight_layout(); plt.savefig("reports/figures/d8_spectra_check.png", dpi=150); plt.close()
# per-band NaN + mean after cleaning
plt.figure(figsize=(10, 3.5))
plt.plot(wl_c, np.isnan(cube).reshape(cube.shape[0], -1).mean(axis=1) * 100)
plt.xlabel("Wavelength (nm)"); plt.ylabel("NaN %"); plt.title("NaN % per band (final)")
plt.tight_layout(); plt.savefig("reports/figures/d8_nan_per_band.png", dpi=150); plt.close()

np.save("data/interim/cleaned_cube.npy", cube.astype("float32"))
np.save("data/interim/wavelengths_clean.npy", wl_c)
pd.DataFrame({"band": np.arange(len(wl_c)), "wavelength_nm": wl_c}).to_csv(
    "data/metadata/wavelengths_clean.csv", index=False)
json.dump(report, open("data/metadata/quality_report.json", "w"), indent=2)
# ============================================================
# DAY 9: reprojection + resampling
# ============================================================
print("\n===== DAY 9: Reprojection / resampling =====")
B, H, W = cube.shape
if sensor == "EMIT":
    src_crs, res = "EPSG:32645", 60.0
    src_tf = from_origin(500000, 2840000, res, res)
else:
    src_crs, res = "EPSG:32616", 20.0
    src_tf = from_origin(500000, 4480000, res, res)
print(f"NOTE: placeholder georeference used (CRS {src_crs}, {res} m). Real scenes: use the scene's own CRS/transform.")

src_path = "data/interim/cleaned_native.tif"
with rasterio.open(src_path, "w", driver="GTiff", height=H, width=W, count=B, dtype="float32",
                   crs=src_crs, transform=src_tf, nodata=np.nan, compress="deflate") as dst:
    dst.write(cube.astype("float32"))
    for i in range(B):
        dst.set_band_description(i + 1, f"{wl_c[i]:.1f}nm")

dst_crs = PP["target_crs"]
tres = PP["target_resolution"]
with rasterio.open(src_path) as src:
    dtf, dw, dh = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=tres)
    out = np.full((B, dh, dw), np.nan, dtype="float32")
    for i in range(B):
        reproject(source=src.read(i + 1), destination=out[i],
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dtf, dst_crs=dst_crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)

out_path = "data/interim/cleaned_reprojected.tif"
with rasterio.open(out_path, "w", driver="GTiff", height=dh, width=dw, count=B, dtype="float32",
                   crs=dst_crs, transform=dtf, nodata=np.nan, compress="deflate") as dst:
    dst.write(out)
    for i in range(B):
        dst.set_band_description(i + 1, f"{wl_c[i]:.1f}nm")
print(f"Saved {out_path} | CRS {dst_crs} | shape (B,H,W) = {out.shape} | pixel size {tres} deg")
# labels: nearest-neighbour reproject (never bilinear for classes)
lab_info = "no labels"
if gt is not None:
    gt_out = np.zeros((dh, dw), dtype="int32")
    reproject(source=gt.astype("int32"), destination=gt_out, src_transform=src_tf, src_crs=src_crs,
              dst_transform=dtf, dst_crs=dst_crs, resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
    np.save("data/interim/gt_reprojected.npy", gt_out)
    assert gt_out.shape == out.shape[1:], "label/image shape mismatch"
    lab_info = f"labels reprojected, shape {gt_out.shape}"
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    idx_g = int(np.argmin(np.abs(wl_c - 800)))
    ax[0].imshow(out[idx_g], cmap="gray"); ax[0].set_title("Reprojected image (~800nm)")
    ax[1].imshow(out[idx_g], cmap="gray"); ax[1].imshow(np.ma.masked_equal(gt_out, 0), cmap="nipy_spectral", alpha=0.5)
    ax[1].set_title("Labels overlay (alignment check)")
    plt.tight_layout(); plt.savefig("reports/figures/d9_alignment_check.png", dpi=150); plt.close()
print(lab_info)
# equivalent GDAL command for the report
with open("scripts/reproject_command.txt", "w") as f:
    f.write(f"gdalwarp -t_srs {dst_crs} -tr {tres} {tres} -r bilinear -srcnodata nan -dstnodata nan "
            f"data/interim/cleaned_native.tif data/interim/cleaned_reprojected_gdal.tif\n")

with open("docs/day7_9_summary.md", "w", encoding="utf-8") as f:
    f.write(f"# Day 7-9 Summary ({date.today()})\n\n- Dataset: {D['name']}\n")
    for k, v in report.items():
        f.write(f"- {k}: {v}\n")
    f.write(f"- Reprojected to {dst_crs} at {tres} deg, bilinear (labels: nearest)\n")
    f.write("- Georeference was a PLACEHOLDER for test data; replace with real scene CRS/transform\n")
# ============================================================
# Tracker + Git
# ============================================================
print("\n===== Tracker + Git =====")
if os.path.exists("scripts/mark_done.py"):
    for d in ("7", "8"):
        subprocess.run([sys.executable, "scripts/mark_done.py", d, "3"], stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "scripts/mark_done.py", "9", "3"])
try:
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Day 7-9: bad bands, masking, gap fill, quality, reprojection"], check=True)
    if subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip():
        subprocess.run(["git", "push"], check=False)
except Exception as e:
    print("Git issue:", e)

print("\nDAY 7-9 DONE. Check reports/figures/d7_*, d8_*, d9_* and docs/day7_9_summary.md")