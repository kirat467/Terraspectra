"""Day 24: demo app + Word project report + submission checklist + week 4 write-up."""
import os, sys, json, glob, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

AUTHOR = "Kailash Kumar"
REF_NO = "ITS/DSML/1513"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit", "python-docx", "pandas", "h5py", "matplotlib"])

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

os.makedirs("reports", exist_ok=True)
os.makedirs("docs", exist_ok=True)


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def cs(p):
    return pd.read_csv(p) if os.path.exists(p) else None
# ============================================================
# 1. DEMO APP (Streamlit)
# ============================================================
APP = '''"""TerraSpectra data explorer. Run: streamlit run scripts/demo_app.py"""
import json, os
import numpy as np
import pandas as pd
import h5py
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H5 = "data/processed/terraspectra_dataset.h5"
st.set_page_config(page_title="TerraSpectra Data Explorer", layout="wide")
st.title("TerraSpectra: Hyperspectral Crop Disease Forecasting")
st.caption("Data Collection & Preparation | Kailash Kumar | Infotact Solutions")

if not os.path.exists(H5):
    st.error("Dataset not found. Run the pipeline scripts first.")
    st.stop()


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}


meta = jl("data/metadata/dataset_metadata.json")
scene = jl("data/metadata/scene_metadata.json")
classes = {int(k): v for k, v in meta.get("classes", {}).items()}
with h5py.File(H5, "r") as f:
    wl = f["wavelengths_nm"][:]
    attrs = {k: f.attrs[k] for k in f.attrs}
    sizes = {s: int(f[s]["y"].shape[0]) for s in ("train", "val", "test")}


@st.cache_data
def labels_of(split):
    with h5py.File(H5, "r") as f:
        return f[split]["y"][:]


@st.cache_data
def class_spectra(max_patches=1500):
    out = {}
    with h5py.File(H5, "r") as f:
        g = f["train"]
        y = g["y"][:]
        for c in np.unique(y):
            idx = np.where(y == c)[0][:max_patches]
            idx = np.sort(idx)
            X = g["X"][idx.tolist()]
            vm = g["valid_mask"][idx.tolist()][:, None].astype("float32")
            out[int(c)] = ((X * vm).sum(axis=(0, 2, 3)) / (vm.sum() + 1e-8)).astype("float32")
    return out


def stretch(a):
    lo, hi = np.percentile(a, (2, 98))
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


tabs = st.tabs(["Overview", "Class spectra", "Patch viewer", "Weather", "Baselines"])
