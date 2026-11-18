import os, sys, json, glob, ast, hashlib, shutil, subprocess, zipfile
from datetime import date
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (change here) =====================
RUN_REPRO_CHECK = True          # re-runs day16_17_18_all.py and compares file hashes (takes a minute)
AUTHOR = "Kailash Kumar"
REF_NO = "ITS/DSML/1513"
# ==================================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-pptx", "h5py", "pandas"])

import numpy as np
import pandas as pd
import h5py
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

for d in ("docs", "reports", "outputs", "handover"):
    os.makedirs(d, exist_ok=True)


def jl(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else (default if default is not None else {})


def md_table(df):
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    cols = [str(c) for c in df.columns]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(str(v) for v in r.values) + " |\n"
    return out


def mark(days, hours="3"):
    if os.path.exists("scripts/mark_done.py"):
        for d in days:
            subprocess.run([sys.executable, "scripts/mark_done.py", str(d), hours], stdout=subprocess.DEVNULL)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


H5 = "data/processed/terraspectra_dataset.h5"
if not os.path.exists(H5):
    raise SystemExit("Final dataset not found. Run day16_17_18_all.py first.")
