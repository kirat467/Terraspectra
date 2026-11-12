import os, sys, csv, subprocess, urllib.request
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (yahan badlo) =====================
BBOX = (84.0, 25.0, 86.0, 26.5)          # (west, south, east, north) - apna region daalo
DATE_RANGE = ("2024-01-01", "2025-12-31")
# ==================================================================