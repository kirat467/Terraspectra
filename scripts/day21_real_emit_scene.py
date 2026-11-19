"""Day 21: real EMIT scene -> orthorectified, officially masked crop for the pipeline."""
import os, sys, glob, json, re, shutil, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (change here) =====================
BBOX = (84.0, 25.0, 86.0, 26.5)          # (west, south, east, north)
DATE_RANGE = ("2024-01-01", "2025-12-31")
CENTER_LAT, CENTER_LON = 25.75, 85.0     # crop centre (falls back to scene centroid if outside)
CROP = 512                               # crop size in pixels (512 ~ 30 km at 60 m)
AUTO_PICK_LOWEST_CLOUD = True
SCENE_INDEX = 0                          # used only if AUTO_PICK_LOWEST_CLOUD = False
MASK_WATER = False                       # True = also mask water pixels
ARCHIVE_OLD = True                       # move Indian Pines outputs to data/archive_previous_run
SAVE_TIF = True                          # also save a GeoTIFF (open in QGIS)
# ==================================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "earthaccess", "xarray", "netCDF4", "affine"])

import numpy as np
import xarray as xr
import earthaccess
from affine import Affine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for d in ("data/raw/emit_real", "data/interim", "data/metadata", "data/labels", "reports/figures"):
    os.makedirs(d, exist_ok=True)
# ---------- 1. search + pick scene ----------
earthaccess.login()
res = earthaccess.search_data(short_name="EMITL2ARFL", bounding_box=BBOX, temporal=DATE_RANGE, count=30)
if not res:
    raise SystemExit("No EMIT scenes found. Widen BBOX / DATE_RANGE (EMIT coverage is sparse).")


def cloud_of(r):
    try:
        return float(r["umm"].get("CloudCover"))
    except Exception:
        return 999.0


print(f"{len(res)} scenes found:")
for i, r in enumerate(res):
    print(f"  [{i}] {r['umm']['GranuleUR']} | cloud: {cloud_of(r) if cloud_of(r) < 999 else 'NA'}")
k = int(np.argmin([cloud_of(r) for r in res])) if AUTO_PICK_LOWEST_CLOUD else SCENE_INDEX
granule = res[k]["umm"]["GranuleUR"]
print("Selected:", granule)
# ---------- 2. download only reflectance + mask ----------
links = [l for l in res[k].data_links() if l.endswith(".nc") and "UNCERT" not in l.upper()]
print("Downloading:", [os.path.basename(l) for l in links], "(can be a few GB)")
earthaccess.download(links, "data/raw/emit_real")
nc = glob.glob("data/raw/emit_real/*.nc")
rfl_files = [f for f in nc if "_RFL_" in os.path.basename(f) and "UNCERT" not in f.upper()]
mask_files = [f for f in nc if "_MASK_" in os.path.basename(f)]
if not rfl_files:
    raise SystemExit("Reflectance file not found in data/raw/emit_real. Download may have failed.")
rfl_file = rfl_files[0]
mask_file = mask_files[0] if mask_files else None
print("Reflectance:", os.path.basename(rfl_file), "| Mask:", os.path.basename(mask_file) if mask_file else "NOT FOUND")