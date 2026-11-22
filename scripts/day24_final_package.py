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
# ============================================================
# 2. WORD PROJECT REPORT
# ============================================================
scene = jl("data/metadata/scene_metadata.json")
quality = jl("data/metadata/quality_report.json")
smooth = jl("data/metadata/smoothing_choice.json")
dsm = jl("data/metadata/dataset_metadata.json")
cw = jl("data/processed/class_weights.json")
leak = jl("data/metadata/leakage_demo.json")
base = cs("data/metadata/baseline_results.csv")
fisher = cs("data/metadata/indices_fisher_scores.csv")
dist = cs("data/metadata/class_distribution_patches.csv")
lstats = cs("data/metadata/label_stats.csv")
top_b = cs("data/metadata/top_bands.csv")
dataset_name = scene.get("dataset", "unknown")
is_test = "Indian" in dataset_name
n_p = dsm.get("n_patches", {})
idx_names = dsm.get("indices", [])

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)


def table(df, max_rows=25):
    if df is None or len(df) == 0:
        return
    df = df.head(max_rows)
    t = doc.add_table(rows=1, cols=len(df.columns))
    try:
        t.style = "Light Grid Accent 1"
    except Exception:
        t.style = "Table Grid"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, r in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(r.values):
            cells[i].text = f"{v:.4g}" if isinstance(v, float) else str(v)
    doc.add_paragraph()


def fig(name, caption, width=6.0):
    p = f"reports/figures/{name}.png"
    if os.path.exists(p):
        doc.add_picture(p, width=Inches(width))
        c = doc.add_paragraph(caption)
        c.runs[0].italic = True


def bullets(items):
    for x in items:
        doc.add_paragraph(str(x), style="List Bullet")


doc.add_heading("TerraSpectra: Hyperspectral Crop Disease Forecasting", 0)
doc.add_paragraph(f"Project Report: Data Collection & Preparation\n{AUTHOR} | Data Associate L1 Intern | Infotact Solutions\n"
                  f"Ref: {REF_NO} | Date: {date.today()}")

doc.add_heading("1. Objective", 1)
doc.add_paragraph("Build a clean, reproducible and leakage-free hyperspectral dataset (patches, labels, vegetation indices and weather "
                  "features) that the modeling team can use for crop disease detection and forecasting.")

doc.add_heading("2. Data", 1)
bullets([f"Dataset used: {dataset_name}",
         f"Scene shape (bands, H, W): {scene.get('shape_BHW')}",
         f"Wavelength range: {scene.get('wavelength_min_nm')} - {scene.get('wavelength_max_nm')} nm",
         "Weather: NASA POWER daily data (temperature, humidity, rainfall, radiation)"])
if is_test:
    doc.add_paragraph("Note: the pipeline was developed and validated on the Indian Pines benchmark (crop-type classes, not disease labels) "
                      "while real disease labels were being arranged. All steps are config-driven and can be re-run on the real scene.")

doc.add_heading("3. Cleaning and quality", 1)
table(pd.DataFrame({"item": list(quality.keys()), "value": list(quality.values())}))
doc.add_paragraph("Steps: reflectance scaling, water-absorption band removal, noisy-band detection, cloud/shadow masking, spectral gap filling, clipping to [0, 1].")
fig("d7_bad_band_removal", "Figure 1: Band removal")
fig("d8_spectra_check", "Figure 2: Spectra after cleaning")

doc.add_heading("4. Reprojection and label alignment", 1)
doc.add_paragraph("Imagery was resampled bilinearly and labels with nearest-neighbour so class values are never interpolated. Alignment was verified by overlay.")
fig("d9_alignment_check", "Figure 3: Label alignment check")
if lstats is not None:
    table(lstats)

doc.add_heading("5. Smoothing, normalization and features", 1)
bullets([f"Savitzky-Golay smoothing: window {smooth.get('savgol_window')}, polynomial order {smooth.get('savgol_polyorder')}",
         "Normalization: per-band z-score using training-patch statistics only (prevents leakage)",
         f"Vegetation indices: {idx_names}",
         "Weather risk features: rolling temperature, humidity, rainfall, favourable-day counts (generic fungal-risk proxy)"])
fig("d12_smoothing", "Figure 4: Spectral smoothing")
fig("d13_index_maps", "Figure 5: Vegetation index maps")
table(fisher)

doc.add_heading("6. Dataset preparation", 1)
bullets([f"Patch size {dsm.get('patch_size')}x{dsm.get('patch_size')}, non-overlapping",
         f"Spatial block split (block {dsm.get('block_size_patches')}x{dsm.get('block_size_patches')} patches, seed {dsm.get('split_seed')}); random pixel split avoided",
         f"Patches: train {n_p.get('train')}, validation {n_p.get('val')}, test {n_p.get('test')}",
         f"Classes: {len(dsm.get('classes', {}))}; training imbalance ratio {cw.get('imbalance_ratio_train', 'NA')}",
         "Imbalance handling: balanced class weights and an oversampling index (no data duplicated on disk)"])
fig("d17_spatial_split", "Figure 6: Spatial split map")
fig("d18_class_distribution_splits", "Figure 7: Class distribution per split")
table(dist)

doc.add_heading("7. Dataset validation with baseline models", 1)
doc.add_paragraph("Simple baselines were trained only to confirm the dataset carries learnable signal; they are not the project's final model.")
table(base)
if leak:
    doc.add_paragraph(f"Leakage check: random-split accuracy {leak['random_split']['accuracy']:.3f} vs spatial-split accuracy "
                      f"{leak['spatial_split']['accuracy']:.3f}.")
fig("d23_leakage_demo", "Figure 8: Random vs spatial split")
fig("d23_band_importance", "Figure 9: Important spectral regions")
if top_b is not None:
    table(top_b)

doc.add_heading("8. Limitations and risks", 1)
lim = []
if is_test:
    lim.append("Results are on a benchmark dataset; disease labels for a real scene are the main pending dependency.")
lim += ["Cloud/shadow masking is heuristic unless the official sensor mask is used.",
        "Weather thresholds are a generic proxy and need disease-specific values from a domain expert.",
        "Small test sets give wide confidence intervals; scores should be reported with n_test."]
if cw.get("rare_classes_lt5_train"):
    lim.append(f"Classes with fewer than 5 training patches: {cw['rare_classes_lt5_train']}.")
bullets(lim)

doc.add_heading("9. Reproduction and handover", 1)
bullets(["pip install -r requirements.txt",
         "Run scripts/day4_5_6_all.py through scripts/day19_20_all.py in order (fixed seeds)",
         "Dataset: data/processed/terraspectra_dataset.h5 | Loader: handover/load_example.py",
         "Documentation: data_dictionary.md, dataset_card.md, preprocessing_report.md",
         "Demo: streamlit run scripts/demo_app.py"])

doc.add_heading("10. Conclusion", 1)
doc.add_paragraph("A documented, scripted and leakage-aware data preparation pipeline is in place. The dataset is ready for the modeling "
                  "team, and the same scripts can process a real hyperspectral scene once disease labels are available.")
rp = "reports/TerraSpectra_Project_Report.docx"
doc.save(rp)
print("Saved", rp)
# ============================================================
# 3. SUBMISSION CHECKLIST
# ============================================================
tags = subprocess.run(["git", "tag"], capture_output=True, text=True).stdout.split()
items = [
    ("Source code", bool(glob.glob("src/*.py")) and bool(glob.glob("scripts/*.py")), "src/ + scripts/"),
    ("Project documentation", os.path.exists("docs/data_dictionary.md") and os.path.exists("docs/dataset_card.md"), "docs/data_dictionary.md, dataset_card.md"),
    ("Project report", os.path.exists(rp), rp),
    ("Presentation (PPT)", os.path.exists("reports/TerraSpectra_DataPrep_Section.pptx"), "reports/TerraSpectra_DataPrep_Section.pptx (Day 20)"),
    ("Model evaluation results", os.path.exists("docs/baseline_results.md"), "docs/baseline_results.md (Day 23)"),
    ("Final project demonstration", os.path.exists("scripts/demo_app.py"), "streamlit run scripts/demo_app.py"),
    ("Dataset handover package", os.path.exists("outputs/TerraSpectra_Dataset_Handover.zip"), "outputs/TerraSpectra_Dataset_Handover.zip"),
    ("Final dataset (HDF5)", os.path.exists("data/processed/terraspectra_dataset.h5"), "data/processed/terraspectra_dataset.h5"),
    ("Git tag v1.0-dataset", "v1.0-dataset" in tags, "git tag"),
    ("Real scene + real disease labels", os.path.exists("data/labels/field_points.csv") and not is_test,
     "PENDING: confirm label source with mentor" if is_test else "data/labels/field_points.csv"),
]
with open("docs/submission_checklist.md", "w", encoding="utf-8") as f:
    f.write(f"# Submission checklist ({date.today()})\n\n")
    for n, ok, where in items:
        f.write(f"- [{'x' if ok else ' '}] {n}: {where}\n")
print("\n===== SUBMISSION CHECKLIST =====")
for n, ok, where in items:
    print(f"[{'OK' if ok else '--'}] {n}: {where}")
    # ============================================================
# 4. WEEK 4 WRITE-UP
# ============================================================
rf = base[(base["model"] == "random_forest") & (base["split"] == "test")] if base is not None else None
maj = base[(base["model"] == "majority_class") & (base["split"] == "test")] if base is not None else None
lines = ["During Week 4, I focused on validating the dataset, documentation, and final delivery for the TerraSpectra project.", "", "Work completed:"]
lines += ["- Completed the final HDF5 dataset with a data dictionary, dataset card and preprocessing report.",
          "- Verified reproducibility by re-running the split pipeline with fixed seeds and comparing file hashes.",
          "- Prepared the handover package (dataset, normalization statistics, class weights, loader example) for the ML team."]
if rf is not None and len(rf):
    lines.append(f"- Validated the dataset with baseline models (Random Forest test accuracy {float(rf['accuracy'].iloc[0]):.3f}, "
                 f"macro-F1 {float(rf['macro_f1'].iloc[0]):.3f}; majority-class baseline {float(maj['accuracy'].iloc[0]):.3f}).")
if leak:
    lines.append(f"- Demonstrated why spatial splitting matters (random split {leak['random_split']['accuracy']:.2f} vs spatial split {leak['spatial_split']['accuracy']:.2f}).")
lines += ["- Built the final project report (Word), a PPT section and an interactive Streamlit demo of the dataset.",
          "- Pushed all code and documentation to the Member-3 branch on GitHub."]
if is_test:
    lines += ["", "The pipeline was validated on the Indian Pines benchmark; real disease labels remain the main pending dependency, which I am discussing with the mentor."]
open("docs/week4_writeup.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\nSaved docs/week4_writeup.txt (paste into the weekly form after reading it)")

try:
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "Day 24: demo app, Word report, submission checklist"], check=False)
    if subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip():
        subprocess.run(["git", "push"], check=False)
except Exception as e:
    print("Git issue:", e)

print("\nDAY 24 DONE. Demo: streamlit run scripts/demo_app.py | Report: reports/TerraSpectra_Project_Report.docx")