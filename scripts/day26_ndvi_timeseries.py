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