import os, sys, json, subprocess, urllib.request
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

# ===================== SETTINGS (change here) =====================
LAT, LON = 25.75, 85.0                    # centre of your study region
WEATHER_START, WEATHER_END = "20250601", "20250930"   # YYYYMMDD (use your scene dates)
MIN_CLASS_PIXELS = 50                     # classes smaller than this get flagged
# ==================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
os.makedirs("data/labels", exist_ok=True)


def mark(days, hours="3"):
    if os.path.exists("scripts/mark_done.py"):
        for d in days:
            subprocess.run([sys.executable, "scripts/mark_done.py", str(d), hours], stdout=subprocess.DEVNULL)


def git(msg):
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", msg], check=False)
        if subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip():
            subprocess.run(["git", "push"], check=False)
    except Exception as e:
        print("Git issue:", e)
        # ============================================================
# Load inputs from Day 12
# ============================================================
need = ["data/interim/smoothed_reflectance.npy", "data/interim/wavelengths_clean.npy"]
for p in need:
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found. Run day10_11_12_all.py first.")
refl = np.load(need[0]).astype("float32")          # (B,H,W) smoothed reflectance 0-1
wl = np.load(need[1])
B, H, W = refl.shape
labels_raw = np.load("data/interim/preprocessed_labels.npy") if os.path.exists("data/interim/preprocessed_labels.npy") else None
is_test = labels_raw is not None and os.path.exists("data/interim/indian_pines_gt.npy")
print(f"Reflectance cube: {refl.shape} | wavelengths {wl.min():.0f}-{wl.max():.0f} nm")
if is_test:
    print("NOTE: Indian Pines wavelengths are approximate, so index VALUES are for pipeline testing only.")

nan_px = np.isnan(refl).any(axis=0)