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