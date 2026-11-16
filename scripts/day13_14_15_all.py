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