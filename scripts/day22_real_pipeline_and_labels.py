"""Day 22: patch pipeline for the real EMIT scene, run it, build label helpers, ingest labels."""
import os, sys, re, json, glob, shutil, subprocess
from datetime import datetime, timedelta, date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (change here) =====================
RUN_PIPELINE = True          # run day7_8_9 -> day10_11_12 -> day13_14_15 on the real scene
RUN_AFTER_LABELS = True      # if valid labels exist, also run day16_17_18 and day19_20
K_ZONES = 5                  # unsupervised spectral zones
N_CANDIDATES = 30            # field-visit points (half stress candidates, half healthy references)
STRESS_Z = -1.5              # stress candidate = vegetation score below this z-score
MIN_POINTS_PER_CLASS = 30    # minimum usable label points per class
# ==================================================================