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
with tabs[0]:
    st.subheader("Dataset summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train patches", sizes["train"]); c2.metric("Val patches", sizes["val"])
    c3.metric("Test patches", sizes["test"]); c4.metric("Classes", len(classes))
    st.write(f"Source data: **{scene.get('dataset', 'unknown')}** | bands: {len(wl)} | "
             f"wavelengths {wl.min():.0f}-{wl.max():.0f} nm | patch size {attrs.get('patch_size')}")
    rows = []
    for s in ("train", "val", "test"):
        cnt = np.bincount(labels_of(s), minlength=max(len(classes), 1))
        for c, n in enumerate(cnt):
            rows.append({"split": s, "class": classes.get(c, f"class_{c}"), "patches": int(n)})
    st.bar_chart(pd.DataFrame(rows).pivot(index="class", columns="split", values="patches"))
    for fig in ("d17_spatial_split", "d15_label_alignment"):
        p = f"reports/figures/{fig}.png"
        if os.path.exists(p):
            st.image(p, caption=fig)

with tabs[1]:
    st.subheader("Mean normalized spectrum per class (train)")
    spec = class_spectra()
    pick = st.multiselect("Classes", sorted(spec), default=sorted(spec)[:6],
                          format_func=lambda c: classes.get(c, f"class_{c}"))
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in pick:
        ax.plot(wl, spec[c], label=classes.get(c, f"class_{c}"))
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("z-score"); ax.legend(fontsize=7, ncol=3)
    st.pyplot(fig)

with tabs[2]:
    st.subheader("Patch viewer")
    split = st.selectbox("Split", ["train", "val", "test"])
    y = labels_of(split)
    cls = st.selectbox("Class", sorted(np.unique(y)), format_func=lambda c: classes.get(c, f"class_{c}"))
    ids = np.where(y == cls)[0]
    k = st.slider("Patch", 0, max(len(ids) - 1, 0), 0)
    i = int(ids[k])
    with h5py.File(H5, "r") as f:
        g = f[split]
        X, ypix, vm = g["X"][i], g["y_pixel"][i], g["valid_mask"][i]
        Xi = g["X_indices"][i] if "X_indices" in g else None
    b = lambda nm: int(np.argmin(np.abs(wl - nm)))
    fc = np.dstack([stretch(X[b(850)]), stretch(X[b(650)]), stretch(X[b(550)])])
    cols = st.columns(3 if Xi is None else 4)
    cols[0].image(fc, caption="False colour (NIR-R-G)", clamp=True, use_container_width=True)
    cols[1].image(np.where(ypix[..., None] >= 0, 255, 60).astype("uint8").repeat(3, axis=2),
                  caption="Labelled pixels (white)", use_container_width=True)
    cols[2].image((vm * 255).astype("uint8"), caption="Valid pixel mask", use_container_width=True)
    if Xi is not None:
        names = json.loads(attrs.get("indices", "[]")) if isinstance(attrs.get("indices"), str) else []
        j = st.selectbox("Vegetation index", range(Xi.shape[0]), format_func=lambda q: names[q] if q < len(names) else str(q))
        cols[3].image(stretch(Xi[j]), caption="Index (normalized)", clamp=True, use_container_width=True)

with tabs[3]:
    p = "data/metadata/weather_daily.csv"
    if os.path.exists(p):
        w = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        st.line_chart(w[["temp_mean", "humidity"]])
        st.bar_chart(w["rain_mm"])
        if "favourable_7d" in w:
            st.line_chart(w["favourable_7d"])
            st.caption("favourable_7d = days in last 7 with 15-30 C and humidity > 80% (generic fungal-risk proxy)")
    else:
        st.info("weather_daily.csv not found.")

with tabs[4]:
    p = "data/metadata/baseline_results.csv"
    if os.path.exists(p):
        st.dataframe(pd.read_csv(p))
        lk = jl("data/metadata/leakage_demo.json")
        if lk:
            st.write(f"Random split accuracy {lk['random_split']['accuracy']:.3f} vs spatial split "
                     f"{lk['spatial_split']['accuracy']:.3f}")
        for fig in ("d23_leakage_demo", "d23_band_importance", "d23_confusion_matrix"):
            q = f"reports/figures/{fig}.png"
            if os.path.exists(q):
                st.image(q, caption=fig)
        st.caption("Baselines only validate the dataset; they are not the project's final model.")
    else:
        st.info("Run day23_baseline_validation.py first.")
'''
open("scripts/demo_app.py", "w", encoding="utf-8").write(APP)
print("Saved scripts/demo_app.py  ->  run:  streamlit run scripts/demo_app.py")