"""Day 26: Sentinel-2 NDVI time series + weather lags -> forecasting-ready table."""
import os, sys, json, subprocess, urllib.request
from datetime import date, datetime, timedelta

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (change here) =====================
DATE_START = "2025-11-01"      # crop season start (YYYY-MM-DD)
DATE_END = "2026-04-30"        # crop season end
AOI_KM = 10                    # study box size around the centre (km)
GRID = 5                       # GRID x GRID cells -> one time series per cell
PIX = 250                      # pixels per side when reading the AOI (must be divisible by GRID)
MAX_CLOUD = 40                 # scene-level cloud cover filter (%)
MAX_SCENES = 40                # cap on scenes read (evenly spaced over the season)
MIN_VALID_FRAC = 0.30          # a cell needs this fraction of clear pixels on a date
DROP_THRESH = 0.10             # next-obs NDVI drop bigger than this = "stress proxy"
CENTER_LAT, CENTER_LON = 25.75, 85.0    # used if no real scene metadata exists
# ==================================================================
assert PIX % GRID == 0, "PIX must be divisible by GRID"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pystac-client", "planetary-computer",
                "rasterio", "pandas", "matplotlib", "scipy"])

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.enums import Resampling
import planetary_computer as pc
from pystac_client import Client
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for d in ("data/processed", "data/metadata", "reports/figures", "docs"):
    os.makedirs(d, exist_ok=True)
# ---------- AOI ----------
lat0, lon0 = CENTER_LAT, CENTER_LON
if os.path.exists("data/interim/real_scene_meta.json"):
    bb = json.load(open("data/interim/real_scene_meta.json"))["bbox_lonlat"]
    lon0, lat0 = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    print(f"AOI centre taken from real EMIT scene: {lat0:.4f}, {lon0:.4f}")
dlat = (AOI_KM / 2) / 111.0
dlon = (AOI_KM / 2) / (111.0 * np.cos(np.radians(lat0)))
aoi = [lon0 - dlon, lat0 - dlat, lon0 + dlon, lat0 + dlat]
print("AOI (lon/lat):", [round(v, 4) for v in aoi], "|", DATE_START, "->", DATE_END)