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