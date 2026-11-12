import os, sys, csv, subprocess, urllib.request
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (yahan badlo) =====================
BBOX = (84.0, 25.0, 86.0, 26.5)          # (west, south, east, north) - apna region daalo
DATE_RANGE = ("2024-01-01", "2025-12-31")
# ==================================================================
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log_source(name, n_scenes, note):
    path = "docs/source_log.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "source", "n_scenes", "note"])
        w.writerow([date.today(), name, n_scenes, note])
