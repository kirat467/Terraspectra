import os, sys, json, subprocess, urllib.request
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

# ===================== SETTINGS (change here) =====================
LAT, LON = 25.75, 85.0                    # centre of your study region
WEATHER_START, WEATHER_END = "20250601", "20250930"   # YYYYMMDD (use your scene dates)
MIN_CLASS_PIXELS = 50                     # classes smaller than this get flagged
# ==================================================================
