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