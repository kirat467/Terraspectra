import os, sys, json, hashlib, glob, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (yahan badlo) =====================
DOWNLOAD_EMIT = False                      # True = ek real EMIT scene download hogi (~1-2 GB)
BBOX = (84.0, 25.0, 86.0, 26.5)            # (west, south, east, north)
DATE_RANGE = ("2024-01-01", "2025-12-31")
CROP_SIZE = 512                            # badi scene ka itna chhota subset EDA ke liye
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "xarray", "netCDF4", "earthaccess"])

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)
os.makedirs("data/raw/emit", exist_ok=True)


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# ============================================================
# DAY 4: Download + inventory
# ============================================================
print("\n===== DAY 4: Data download + inventory =====")

if DOWNLOAD_EMIT:
    try:
        import earthaccess
        earthaccess.login()
        res = earthaccess.search_data(short_name="EMITL2ARFL", bounding_box=BBOX,
                                      temporal=DATE_RANGE, count=5)
        print("Scenes mile:", len(res))
        if res:
            earthaccess.download(res[:1], "data/raw/emit")
    except Exception as e:
        print("EMIT download fail:", e)

rows = []
for p in glob.glob("data/raw/**/*", recursive=True):
    if os.path.isfile(p) and not p.endswith(".gitkeep"):
        size_mb = os.path.getsize(p) / 1e6
        rows.append({"file": p.replace("\\", "/"), "size_MB": round(size_mb, 2),
                     "md5": md5(p) if size_mb < 500 else "skipped(large)"})
inv = pd.DataFrame(rows)
inv.to_csv("data/metadata/raw_inventory.csv", index=False)
print(inv.to_string(index=False) if len(inv) else "data/raw me koi file nahi mili")
# ============================================================
# Loader: EMIT .nc ya Indian Pines
# ============================================================


def load_data():
    nc_files = glob.glob("data/raw/emit/*.nc")
    if nc_files:
        import xarray as xr
        p = nc_files[0]
        print("Using EMIT scene:", p)
        ds = xr.open_dataset(p)
        refl = ds["reflectance"]                       # (downtrack, crosstrack, bands)
        bp = xr.open_dataset(p, group="sensor_band_parameters")
        wl = bp["wavelengths"].values.astype("float32")
        good = bp["good_wavelengths"].values.astype(bool) if "good_wavelengths" in bp else np.ones(len(wl), bool)
        H, W = refl.shape[0], refl.shape[1]
        r0, c0 = max(0, H // 2 - CROP_SIZE // 2), max(0, W // 2 - CROP_SIZE // 2)
        sub = refl[r0:r0 + CROP_SIZE, c0:c0 + CROP_SIZE, :].values.astype("float32")
        sub[sub <= -9990] = np.nan
        cube = np.transpose(sub, (2, 0, 1))
        return dict(name=os.path.basename(p), cube=cube, wl=wl, gt=None, good=good,
                    meta={"sensor": "EMIT", "full_shape_HWB": [int(H), int(W), int(refl.shape[2])],
                          "crop_origin_rc": [int(r0), int(c0)], "attrs": {k: str(v) for k, v in list(ds.attrs.items())[:15]}})
    # fallback: Indian Pines
    cp, gp = "data/interim/indian_pines_cube.npy", "data/interim/indian_pines_gt.npy"
    if os.path.exists(cp):
        cube, gt = np.load(cp), np.load(gp)
    else:
        mp = "data/raw/indian_pines/Indian_pines_corrected.mat"
        gpm = "data/raw/indian_pines/Indian_pines_gt.mat"
        if not os.path.exists(mp):
            raise SystemExit("Indian Pines bhi nahi mila. Pehle day2_day3_all.py chalao.")
        cube = np.transpose(sio.loadmat(mp)["indian_pines_corrected"], (2, 0, 1)).astype("float32")
        gt = sio.loadmat(gpm)["indian_pines_gt"].astype("int32")
    wl = np.linspace(400, 2500, cube.shape[0]).astype("float32")   # approx wavelengths
    return dict(name="Indian_Pines", cube=cube.astype("float32"), wl=wl, gt=gt,
                good=np.ones(cube.shape[0], bool),
                meta={"sensor": "AVIRIS (Indian Pines)", "note": "wavelengths approximate (linspace 400-2500nm)"})


D = load_data()
cube, wl, gt, good = D["cube"], D["wl"], D["gt"], D["good"]
B, H, W = cube.shape
# ============================================================
# DAY 5: Metadata inspection
# ============================================================
print("\n===== DAY 5: Metadata inspection =====")
meta = {
    "dataset": D["name"],
    "shape_BHW": [int(B), int(H), int(W)],
    "dtype": str(cube.dtype),
    "wavelength_min_nm": float(wl.min()),
    "wavelength_max_nm": float(wl.max()),
    "avg_band_spacing_nm": float(np.mean(np.diff(wl))),
    "nan_percent": round(float(np.isnan(cube).mean() * 100), 4),
    "value_min": float(np.nanmin(cube)),
    "value_max": float(np.nanmax(cube)),
    "value_mean": float(np.nanmean(cube)),
    "n_good_bands_flagged": int(good.sum()),
    "memory_MB": round(cube.nbytes / 1e6, 1),
}
meta.update(D["meta"])
if gt is not None:
    cl, cn = np.unique(gt, return_counts=True)
    meta["classes"] = {int(a): int(b) for a, b in zip(cl, cn)}
with open("data/metadata/scene_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
pd.DataFrame({"band_index": np.arange(B), "wavelength_nm": wl, "good_band": good}).to_csv(
    "data/metadata/wavelengths.csv", index=False)
print(json.dumps({k: v for k, v in meta.items() if k not in ("attrs",)}, indent=2))
# ============================================================
# DAY 6: EDA
# ============================================================
print("\n===== DAY 6: EDA =====")

# 1) band-wise statistics
flat = cube.reshape(B, -1)
stats = pd.DataFrame({
    "band": np.arange(B), "wavelength_nm": wl,
    "mean": np.nanmean(flat, axis=1), "std": np.nanstd(flat, axis=1),
    "min": np.nanmin(flat, axis=1), "max": np.nanmax(flat, axis=1),
    "nan_pct": np.isnan(flat).mean(axis=1) * 100,
})
stats["snr_proxy"] = stats["mean"].abs() / (stats["std"] + 1e-9)
stats.to_csv("data/metadata/band_stats.csv", index=False)
print("Saved data/metadata/band_stats.csv")

# 2) mean spectrum +/- std
plt.figure(figsize=(9, 4))
plt.plot(wl, stats["mean"], label="mean")
plt.fill_between(wl, stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=0.25, label="±1 std")
plt.xlabel("Wavelength (nm)"); plt.ylabel("Value"); plt.title("Mean spectrum")
plt.legend(); plt.tight_layout(); plt.savefig("reports/figures/eda_mean_spectrum.png", dpi=150); plt.close()

# 3) histograms of 6 sample bands
sample = np.linspace(0, B - 1, 6).astype(int)
fig, ax = plt.subplots(2, 3, figsize=(11, 6))
for a, b in zip(ax.ravel(), sample):
    v = cube[b][~np.isnan(cube[b])]
    a.hist(v, bins=60); a.set_title(f"band {b} ({wl[b]:.0f} nm)")
plt.tight_layout(); plt.savefig("reports/figures/eda_histograms.png", dpi=150); plt.close()

# 4) false-colour composite (NIR, Red, Green)


def nb(nm):
    return int(np.argmin(np.abs(wl - nm)))


def stretch(x):
    lo, hi = np.nanpercentile(x, (2, 98))
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


fc = np.dstack([stretch(cube[nb(850)]), stretch(cube[nb(650)]), stretch(cube[nb(550)])])
rgb = np.dstack([stretch(cube[nb(650)]), stretch(cube[nb(550)]), stretch(cube[nb(470)])])
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
ax[0].imshow(np.nan_to_num(rgb)); ax[0].set_title("Approx RGB")
ax[1].imshow(np.nan_to_num(fc)); ax[1].set_title("False colour (NIR-R-G)")
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.savefig("reports/figures/eda_composites.png", dpi=150); plt.close()

# 5) NaN map + simple NDVI
nan_map = np.isnan(cube).mean(axis=0)
nir, red = cube[nb(800)], cube[nb(670)]
ndvi = (nir - red) / (nir + red + 1e-9)
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
im0 = ax[0].imshow(nan_map, cmap="gray_r"); ax[0].set_title("NaN fraction per pixel"); plt.colorbar(im0, ax=ax[0])
im1 = ax[1].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=1); ax[1].set_title("NDVI (quick)"); plt.colorbar(im1, ax=ax[1])
plt.tight_layout(); plt.savefig("reports/figures/eda_nan_ndvi.png", dpi=150); plt.close()

# 6) class-wise spectra (agar labels hain)
if gt is not None:
    plt.figure(figsize=(10, 5))
    for c in np.unique(gt):
        if c == 0:
            continue
        plt.plot(wl, np.nanmean(cube[:, gt == c], axis=1), label=f"class {c}")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Mean value"); plt.title("Class-wise mean spectra")
    plt.legend(ncol=4, fontsize=7); plt.tight_layout()
    plt.savefig("reports/figures/eda_classwise_spectra.png", dpi=150); plt.close()

# 7) band correlation matrix (sample pixels se)
rng = np.random.default_rng(42)
valid = np.where(~np.isnan(flat).any(axis=0))[0]
pick = rng.choice(valid, size=min(5000, len(valid)), replace=False)
corr = np.corrcoef(flat[:, pick])
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="viridis", vmin=-1, vmax=1); plt.colorbar(label="correlation")
plt.title("Band-to-band correlation"); plt.xlabel("band"); plt.ylabel("band")
plt.tight_layout(); plt.savefig("reports/figures/eda_band_corr.png", dpi=150); plt.close()

# 8) suspicious (noisy) bands
z = (stats["std"] - stats["std"].mean()) / (stats["std"].std() + 1e-9)
noisy = stats.loc[z.abs() > 3, "band"].tolist()
low_snr = stats.nsmallest(10, "snr_proxy")["band"].tolist()
print("Std outlier (noisy) bands:", noisy)
print("10 sabse low-SNR bands:", low_snr)

# summary file
with open("docs/day4_6_summary.md", "w", encoding="utf-8") as f:
    f.write(f"# Day 4-6 Summary ({date.today()})\n\n")
    f.write(f"- Dataset: {D['name']}\n- Shape (B,H,W): {B}, {H}, {W}\n")
    f.write(f"- Wavelength: {wl.min():.0f}-{wl.max():.0f} nm\n- NaN %: {meta['nan_percent']}\n")
    f.write(f"- Noisy bands (std outlier): {noisy}\n- Lowest SNR bands: {low_snr}\n")
    f.write("- Figures: reports/figures/eda_*.png\n- Stats: data/metadata/band_stats.csv\n")
print("Saved docs/day4_6_summary.md")
