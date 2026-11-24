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
# ============================================================
# 1. Search Sentinel-2 L2A scenes
# ============================================================
cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
search = cat.search(collections=["sentinel-2-l2a"], bbox=aoi, datetime=f"{DATE_START}/{DATE_END}",
                    query={"eo:cloud_cover": {"lt": MAX_CLOUD}})
items = list(search.items())
print("Scenes found:", len(items))


def contains(b):
    return b[0] <= aoi[0] and b[1] <= aoi[1] and b[2] >= aoi[2] and b[3] >= aoi[3]


items = [it for it in items if it.bbox and contains(it.bbox)]
best = {}
for it in items:       # one scene per date: lowest cloud cover
    d = it.datetime.date()
    if d not in best or it.properties.get("eo:cloud_cover", 100) < best[d].properties.get("eo:cloud_cover", 100):
        best[d] = it
items = [best[d] for d in sorted(best)]
print("Scenes fully covering the AOI (one per date):", len(items))
if len(items) < 4:
    raise SystemExit("Too few scenes. Widen DATE range, raise MAX_CLOUD or increase AOI_KM.")
if len(items) > MAX_SCENES:
    items = [items[i] for i in np.linspace(0, len(items) - 1, MAX_SCENES).astype(int)]
    print("Subsampled to", len(items), "scenes")
    # ============================================================
# 2. Read AOI window -> NDVI -> grid-cell stats
# ============================================================
def read_window(href, shape, resampling):
    with rasterio.open(href) as src:
        l, b, r, t = transform_bounds("EPSG:4326", src.crs, *aoi, densify_pts=21)
        w = from_bounds(l, b, r, t, transform=src.transform)
        win = Window(int(round(w.col_off)), int(round(w.row_off)),
                     max(1, int(round(w.width))), max(1, int(round(w.height))))
        if win.col_off < 0 or win.row_off < 0 or win.col_off + win.width > src.width or win.row_off + win.height > src.height:
            raise ValueError("AOI window outside this tile")
        return src.read(1, window=win, out_shape=shape, resampling=resampling).astype("float32")


S = PIX // GRID
rows, cubes, used_dates = [], [], []
for n, it in enumerate(items, 1):
    d = it.datetime.date()
    try:
        sit = pc.sign(it)
        b04 = read_window(sit.assets["B04"].href, (PIX, PIX), Resampling.average)
        b08 = read_window(sit.assets["B08"].href, (PIX, PIX), Resampling.average)
        scl = read_window(sit.assets["SCL"].href, (PIX, PIX), Resampling.nearest)
    except Exception as e:
        print(f"  [{n}/{len(items)}] {d} skipped: {str(e)[:80]}")
        continue
    pb = float(str(it.properties.get("s2:processing_baseline", "04.00")))
    off = 1000.0 if pb >= 4.0 else 0.0                      # reflectance offset since baseline 04.00
    red, nir = (b04 - off) / 10000.0, (b08 - off) / 10000.0
    ndvi = np.clip((nir - red) / (nir + red + 1e-8), -1, 1)
    good = np.isin(scl, [4, 5]) & (b04 > 0) & (b08 > 0)     # vegetation / bare soil only (no cloud, shadow, snow, water)
    ndvi = np.where(good, ndvi, np.nan)
    cell_mean = np.full((GRID, GRID), np.nan)
    for gr in range(GRID):
        for gc in range(GRID):
            blk = ndvi[gr * S:(gr + 1) * S, gc * S:(gc + 1) * S]
            vf = float(np.isfinite(blk).mean())
            if vf >= MIN_VALID_FRAC:
                cell_mean[gr, gc] = np.nanmean(blk)
                rows.append({"date": pd.Timestamp(d), "cell": gr * GRID + gc, "row": gr, "col": gc,
                             "ndvi": float(np.nanmean(blk)), "ndvi_std": float(np.nanstd(blk)),
                             "valid_frac": vf, "scene_cloud": it.properties.get("eo:cloud_cover")})
    cubes.append(cell_mean); used_dates.append(pd.Timestamp(d))
    print(f"  [{n}/{len(items)}] {d} cloud {it.properties.get('eo:cloud_cover', 'NA')}% | clear pixels {good.mean() * 100:.0f}% "
          f"| cells with data {int(np.isfinite(cell_mean).sum())}/{GRID * GRID}")

obs = pd.DataFrame(rows)
if obs.empty:
    raise SystemExit("No usable observations. Check network access to planetarycomputer.microsoft.com.")
obs.to_csv("data/processed/ndvi_timeseries_long.csv", index=False)
print(f"\nObservations: {len(obs)} | dates: {obs['date'].nunique()} | cells: {obs['cell'].nunique()}")