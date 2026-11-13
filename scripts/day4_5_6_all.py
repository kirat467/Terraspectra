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
