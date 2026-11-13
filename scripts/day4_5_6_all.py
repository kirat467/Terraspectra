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

