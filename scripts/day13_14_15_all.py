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

# ============================================================
# DAY 13: Vegetation indices
# ============================================================
print("\n===== DAY 13: Vegetation indices =====")


def band(nm, tol=25):
    i = int(np.argmin(np.abs(wl - nm)))
    return i if abs(wl[i] - nm) <= tol else None


def R(nm):
    i = band(nm)
    return None if i is None else refl[i]


def nd(a, b):
    return (a - b) / (a + b + 1e-8)


b445, b480, b531, b570 = R(445), R(480), R(531), R(570)
b670, b680, b705, b720, b750, b790, b800, b860, b1240 = (R(670), R(680), R(705), R(720),
                                                         R(750), R(790), R(800), R(860), R(1240))
indices = {}
if b800 is not None and b670 is not None:
    indices["NDVI"] = nd(b800, b670)
if b531 is not None and b570 is not None:
    indices["PRI"] = nd(b531, b570)
if b750 is not None and b705 is not None:
    indices["RE_NDVI"] = nd(b750, b705)
if b790 is not None and b720 is not None:
    indices["NDRE"] = nd(b790, b720)
if b800 is not None and b670 is not None and b480 is not None:
    indices["EVI"] = np.clip(2.5 * (b800 - b670) / (b800 + 6 * b670 - 7.5 * b480 + 1), -1, 2)
if b860 is not None and b1240 is not None:
    indices["NDWI"] = nd(b860, b1240)
if b800 is not None and b445 is not None and b680 is not None:
    indices["SIPI"] = np.clip((b800 - b445) / (b800 - b680 + 1e-8), -2, 5)

skipped = [n for n in ("NDVI", "PRI", "RE_NDVI", "NDRE", "EVI", "NDWI", "SIPI") if n not in indices]
print("Computed:", list(indices.keys()))
if skipped:
    print("Skipped (needed wavelength not covered by sensor):", skipped)

names = list(indices.keys())
idx_cube = np.stack([indices[n] for n in names]).astype("float32")
idx_cube[:, nan_px] = np.nan
np.save("data/interim/indices_cube.npy", idx_cube)
json.dump(names, open("data/metadata/indices_names.json", "w"))

# stats
st = pd.DataFrame({
    "index": names,
    "mean": [float(np.nanmean(idx_cube[i])) for i in range(len(names))],
    "std": [float(np.nanstd(idx_cube[i])) for i in range(len(names))],
    "min": [float(np.nanmin(idx_cube[i])) for i in range(len(names))],
    "max": [float(np.nanmax(idx_cube[i])) for i in range(len(names))],
})
st.to_csv("data/metadata/indices_stats.csv", index=False)
print(st.round(4).to_string(index=False))

# index maps
n = len(names)
cols = 4
rows = (n + cols - 1) // cols
fig, ax = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.8 * rows))
ax = np.atleast_1d(ax).ravel()
for a in ax:
    a.axis("off")
for k, nm in enumerate(names):
    lo, hi = np.nanpercentile(idx_cube[k], (2, 98))
    im = ax[k].imshow(idx_cube[k], cmap="RdYlGn", vmin=lo, vmax=hi)
    ax[k].set_title(nm); plt.colorbar(im, ax=ax[k], fraction=0.046)
plt.tight_layout(); plt.savefig("reports/figures/d13_index_maps.png", dpi=150); plt.close()

# class separability of indices (needs labels)
if labels_raw is not None and labels_raw.shape == (H, W):
    cls = [c for c in np.unique(labels_raw) if c != 0]
    rows_f = []
    for k, nm in enumerate(names):
        means, vars_, ns = [], [], []
        for c in cls:
            v = idx_cube[k][(labels_raw == c) & ~nan_px]
            if len(v) > 5:
                means.append(v.mean()); vars_.append(v.var()); ns.append(len(v))
        if len(means) > 1:
            means, vars_ = np.array(means), np.array(vars_)
            fisher = np.var(means) / (np.mean(vars_) + 1e-12)
            rows_f.append({"index": nm, "fisher_score": float(fisher)})
    if rows_f:
        fs = pd.DataFrame(rows_f).sort_values("fisher_score", ascending=False)
        fs.to_csv("data/metadata/indices_fisher_scores.csv", index=False)
        print("\nIndex separability (higher = separates classes better):")
        print(fs.round(4).to_string(index=False))
        top = fs.iloc[0]["index"]
        k = names.index(top)
        plt.figure(figsize=(10, 4))
        plt.bar([str(c) for c in cls],
                [np.nanmean(idx_cube[k][(labels_raw == c) & ~nan_px]) if ((labels_raw == c) & ~nan_px).any() else 0 for c in cls])
        plt.xlabel("class"); plt.ylabel(f"mean {top}"); plt.title(f"Class-wise mean of best index: {top}")
        plt.tight_layout(); plt.savefig("reports/figures/d13_best_index_by_class.png", dpi=150); plt.close()
mark([13])

# ============================================================
# DAY 14: Weather data (NASA POWER) for forecasting
# ============================================================
print("\n===== DAY 14: Weather data (NASA POWER) =====")
params = "T2M,T2M_MAX,T2M_MIN,RH2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,WS2M"
url = ("https://power.larc.nasa.gov/api/temporal/daily/point"
       f"?parameters={params}&community=AG&longitude={LON}&latitude={LAT}"
       f"&start={WEATHER_START}&end={WEATHER_END}&format=JSON")
weather = None
try:
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode())
    par = data["properties"]["parameter"]
    weather = pd.DataFrame(par)
    weather.index = pd.to_datetime(weather.index, format="%Y%m%d")
    weather = weather.replace(-999, np.nan)
    weather.columns = ["temp_mean", "temp_max", "temp_min", "humidity", "rain_mm", "solar_rad", "wind"][:len(weather.columns)] \
        if list(weather.columns) == params.split(",") else weather.columns
    weather = weather.rename(columns={"T2M": "temp_mean", "T2M_MAX": "temp_max", "T2M_MIN": "temp_min",
                                      "RH2M": "humidity", "PRECTOTCORR": "rain_mm",
                                      "ALLSKY_SFC_SW_DWN": "solar_rad", "WS2M": "wind"})
    weather = weather.interpolate(limit=3)
except Exception as e:
    print("Weather download failed:", e)
    print("Check internet, then re-run. (Skipping weather - no fake data is generated.)")

if weather is not None and len(weather):
    # disease-risk features (generic fungal proxy: warm + humid + wet)
    weather["temp_7d"] = weather["temp_mean"].rolling(7, min_periods=3).mean()
    weather["humidity_7d"] = weather["humidity"].rolling(7, min_periods=3).mean()
    weather["rain_7d_sum"] = weather["rain_mm"].rolling(7, min_periods=3).sum()
    weather["rain_14d_sum"] = weather["rain_mm"].rolling(14, min_periods=5).sum()
    weather["favourable_day"] = ((weather["temp_mean"].between(15, 30)) & (weather["humidity"] > 80)).astype(int)
    weather["favourable_7d"] = weather["favourable_day"].rolling(7, min_periods=3).sum()
    weather["temp_range"] = weather["temp_max"] - weather["temp_min"]
    weather.index.name = "date"
    weather.to_csv("data/metadata/weather_daily.csv")
    print(f"Saved data/metadata/weather_daily.csv | {len(weather)} days | {weather.index.min().date()} -> {weather.index.max().date()}")
    print(weather[["temp_mean", "humidity", "rain_mm", "favourable_7d"]].describe().round(2).to_string())

    fig, ax = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    ax[0].plot(weather.index, weather["temp_mean"]); ax[0].set_ylabel("Temp (°C)")
    ax[1].plot(weather.index, weather["humidity"], color="tab:green"); ax[1].set_ylabel("Humidity (%)")
    ax[2].bar(weather.index, weather["rain_mm"], color="tab:blue"); ax[2].set_ylabel("Rain (mm)")
    ax[3].plot(weather.index, weather["favourable_7d"], color="tab:red"); ax[3].set_ylabel("Favourable days / 7")
    plt.tight_layout(); plt.savefig("reports/figures/d14_weather.png", dpi=150); plt.close()
mark([14])

# ============================================================
# DAY 15: Label alignment
# ============================================================
print("\n===== DAY 15: Label alignment =====")
INDIAN_PINES_NAMES = {
    1: "Alfalfa", 2: "Corn-notill", 3: "Corn-mintill", 4: "Corn", 5: "Grass-pasture",
    6: "Grass-trees", 7: "Grass-pasture-mowed", 8: "Hay-windrowed", 9: "Oats",
    10: "Soybean-notill", 11: "Soybean-mintill", 12: "Soybean-clean", 13: "Wheat",
    14: "Woods", 15: "Buildings-Grass-Trees-Drives", 16: "Stone-Steel-Towers"}

labels = None
source = ""
# Option A: field points CSV (columns: lat, lon, label) rasterized onto the scene grid
pts_path = "data/labels/field_points.csv"
if os.path.exists(pts_path) and os.path.exists("data/interim/cleaned_reprojected.tif"):
    import rasterio
    from rasterio.transform import rowcol
    pts = pd.read_csv(pts_path)
    with rasterio.open("data/interim/cleaned_reprojected.tif") as s:
        tf, hh, ww = s.transform, s.height, s.width
    labels = np.zeros((hh, ww), dtype="int32")
    for _, r in pts.iterrows():
        rr, cc = rowcol(tf, r["lon"], r["lat"])
        if 0 <= rr < hh and 0 <= cc < ww:
            labels[rr, cc] = int(r["label"])
    source = f"field points CSV ({len(pts)} points)"
# Option B: raster labels from earlier step
elif labels_raw is not None:
    labels = labels_raw.astype("int32")
    source = "labels from Day 9 pipeline (Indian Pines ground truth)" if is_test else "preprocessed_labels.npy"

if labels is None:
    print("No labels found yet. Put field labels in data/labels/field_points.csv (columns: lat,lon,label)")
    print("or wait for the mentor's label source. Creating an empty label map so later steps still run.")
    labels = np.zeros((H, W), dtype="int32")
    source = "EMPTY (no labels yet)"

assert labels.shape == (H, W), f"Label shape {labels.shape} != image shape {(H, W)}. Re-check reprojection."

# 0 = unlabeled -> -1 ; remap classes to contiguous 0..K-1
orig = np.unique(labels[labels > 0])
mapping = {int(o): int(i) for i, o in enumerate(orig)}
aligned = np.full((H, W), -1, dtype="int32")
for o, i in mapping.items():
    aligned[labels == o] = i
# labelled pixels that fall on invalid (NaN) image pixels are unusable
lost = int(((aligned >= 0) & nan_px).sum())
aligned[nan_px] = -1
print(f"Source: {source}")
print(f"Classes: {len(mapping)} | labelled pixels: {int((aligned >= 0).sum())} | lost on NaN pixels: {lost}")

cnt = pd.Series(aligned[aligned >= 0]).value_counts().sort_index()
inv = {v: k for k, v in mapping.items()}
stats = pd.DataFrame({
    "new_label": cnt.index,
    "original_label": [inv[i] for i in cnt.index],
    "class_name": [INDIAN_PINES_NAMES.get(inv[i], f"class_{inv[i]}") if is_test else f"class_{inv[i]}" for i in cnt.index],
    "pixels": cnt.values,
    "percent": (cnt.values / max(cnt.sum(), 1) * 100).round(2),
})
stats["small_class_flag"] = stats["pixels"] < MIN_CLASS_PIXELS
stats.to_csv("data/metadata/label_stats.csv", index=False)
print(stats.to_string(index=False))
if len(stats):
    ratio = stats["pixels"].max() / max(stats["pixels"].min(), 1)
    print(f"Imbalance ratio (max/min): {ratio:.1f}x" + ("  <- high, handle in Day 18" if ratio > 10 else ""))

np.save("data/interim/labels_aligned.npy", aligned)
json.dump({"mapping_original_to_new": mapping, "unlabeled_value": -1},
          open("data/metadata/label_mapping.json", "w"), indent=2)

# overlay check
gi = int(np.argmin(np.abs(wl - 800)))
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].imshow(refl[gi], cmap="gray"); ax[0].set_title("Image (~800 nm)")
ax[1].imshow(np.ma.masked_equal(aligned, -1), cmap="nipy_spectral"); ax[1].set_title("Aligned labels (-1 hidden)")
ax[2].imshow(refl[gi], cmap="gray"); ax[2].imshow(np.ma.masked_equal(aligned, -1), cmap="nipy_spectral", alpha=0.5)
ax[2].set_title("Overlay (alignment check)")
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.savefig("reports/figures/d15_label_alignment.png", dpi=150); plt.close()

if len(stats):
    plt.figure(figsize=(10, 4))
    plt.bar(stats["new_label"].astype(str), stats["pixels"])
    plt.xlabel("class (new label)"); plt.ylabel("pixels"); plt.title("Class distribution")
    plt.tight_layout(); plt.savefig("reports/figures/d15_class_distribution.png", dpi=150); plt.close()

with open("docs/day13_15_summary.md", "w", encoding="utf-8") as f:
    f.write(f"# Day 13-15 Summary ({date.today()})\n\n")
    f.write(f"- Indices computed: {names}\n- Skipped: {skipped}\n")
    f.write(f"- Weather: {'saved' if weather is not None and len(weather) else 'NOT downloaded'} (NASA POWER, lat {LAT}, lon {LON})\n")
    f.write(f"- Labels: {source}; classes {len(mapping)}; labelled pixels {int((aligned >= 0).sum())}\n")
    f.write("- Outputs: indices_cube.npy, labels_aligned.npy, label_mapping.json, weather_daily.csv\n")

mark([15])
git("Day 13-15: vegetation indices, weather data, label alignment")
print("\nDAY 13-15 DONE. Figures: d13_*, d14_weather, d15_* in reports/figures/")