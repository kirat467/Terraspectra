import os, sys, json, hashlib, glob, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (yahan badlo) =====================
DOWNLOAD_EMIT = False                      # True = ek real EMIT scene download hogi (~1-2 GB)
BBOX = (84.0, 25.0, 86.0, 26.5)            # (west, south, east, north)
DATE_RANGE = ("2024-01-01", "2025-12-31")
CROP_SIZE = 512                            # badi scene ka itna chhota subset EDA ke liye