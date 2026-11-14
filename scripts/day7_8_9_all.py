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