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
if not os.path.exists("data/interim/real_scene_meta.json"):
    raise SystemExit("Run day21_real_emit_scene.py first (real_scene_meta.json not found).")
meta = json.load(open("data/interim/real_scene_meta.json"))
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
# ============================================================
# PART A: auto-patch the earlier scripts (backup .bak, safe to re-run)
# ============================================================
def patch(path, old, new, tag, label):
    if not os.path.exists(path):
        print(f"  [{label}] {path} not found"); return False
    s = open(path, encoding="utf-8").read()
    if tag in s:
        print(f"  [{label}] already patched"); return True
    if old not in s:
        print(f"  [{label}] COULD NOT patch (your file text differs). Apply this edit manually."); return False
    if not os.path.exists(path + ".bak"):
        shutil.copy(path, path + ".bak")
    open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print(f"  [{label}] patched"); return True


print("\n===== PART A: patching scripts for the real scene =====")
P789 = "scripts/day7_8_9_all.py"
ok1 = patch(P789,
    'def load_data():\n    nc_files = glob.glob("data/raw/emit/*.nc")',
    'def load_data():\n'
    '    if os.path.exists("data/interim/real_scene_cube.npy"):\n'
    '        print("Using REAL EMIT scene (real_scene_cube.npy)")\n'
    '        return dict(name="EMIT_real", sensor="EMIT",\n'
    '                    cube=np.load("data/interim/real_scene_cube.npy"),\n'
    '                    wl=np.load("data/interim/real_scene_wl.npy"), gt=None,\n'
    '                    good=np.load("data/interim/real_scene_good.npy"))\n'
    '    nc_files = glob.glob("data/raw/emit/*.nc")',
    'real_scene_cube.npy"):', "edit 1: loader")
ok2 = patch(P789,
    'if sensor == "EMIT":\n    def sel(lo, hi):',
    'if sensor == "EMIT" and D["name"] != "EMIT_real":\n    def sel(lo, hi):',
    'D["name"] != "EMIT_real"', "edit 2: skip heuristic cloud mask")
ok3 = patch(P789,
    'if sensor == "EMIT":\n    src_crs, res = "EPSG:32645", 60.0\n    src_tf = from_origin(500000, 2840000, res, res)',
    'if D["name"] == "EMIT_real":\n'
    '    from affine import Affine\n'
    '    _m = json.load(open("data/interim/real_scene_meta.json"))\n'
    '    src_crs, res = "EPSG:4326", 0.0005\n'
    '    src_tf = Affine(*_m["transform"])\n'
    'elif sensor == "EMIT":\n'
    '    src_crs, res = "EPSG:32645", 60.0\n'
    '    src_tf = from_origin(500000, 2840000, res, res)',
    'D["name"] == "EMIT_real"', "edit 3: real georeference")
# config: EMIT native pixel ~0.0005 deg
cfg_p = "config/config.yaml"
s = open(cfg_p, encoding="utf-8").read()
s2 = re.sub(r"target_resolution:\s*[\d.]+", "target_resolution: 0.0005", s)
if s2 != s:
    open(cfg_p, "w", encoding="utf-8").write(s2)
print("  [config] target_resolution = 0.0005")
# weather window + location from the scene
bb = meta["bbox_lonlat"]
c_lon, c_lat = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
sd_txt = meta.get("scene_date", "unknown")
P1315 = "scripts/day13_14_15_all.py"
if os.path.exists(P1315):
    s = open(P1315, encoding="utf-8").read()
    s = re.sub(r"^LAT, LON = .*$", f"LAT, LON = {c_lat:.4f}, {c_lon:.4f}", s, flags=re.M)
    if sd_txt.isdigit():
        sd = datetime.strptime(sd_txt, "%Y%m%d")
        start = (sd - timedelta(days=60)).strftime("%Y%m%d")
        end = min(sd + timedelta(days=7), datetime.now() - timedelta(days=5)).strftime("%Y%m%d")
        s = re.sub(r"^WEATHER_START, WEATHER_END = .*$", f'WEATHER_START, WEATHER_END = "{start}", "{end}"', s, flags=re.M)
        print(f"  [weather] location {c_lat:.3f},{c_lon:.3f} | window {start} -> {end} (60 days before scene date)")
    open(P1315, "w", encoding="utf-8").write(s)

if not (ok1 and ok2 and ok3):
    raise SystemExit("\nSome edits could not be applied automatically. Apply them manually (see previous message), then re-run.")


def run(script):
    print(f"\n>>>>>>>>>> {script}")
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        raise SystemExit(f"\n{script} FAILED. Send me the error shown above.")

# ============================================================
# PART B: run the pipeline on the real scene
# ============================================================
if RUN_PIPELINE:
    if os.path.exists("data/interim/cleaned_reprojected.tif") and os.path.exists("data/interim/smoothed_reflectance.npy"):
        mt = max(os.path.getmtime("data/interim/real_scene_cube.npy"), os.path.getmtime(P789))
        if os.path.getmtime("data/interim/smoothed_reflectance.npy") > mt:
            print("\nPipeline outputs are already newer than the real scene -> skipping Part B (set RUN_PIPELINE=True and delete data/interim/smoothed_reflectance.npy to force)")
        else:
            for sc in ("scripts/day7_8_9_all.py", "scripts/day10_11_12_all.py", "scripts/day13_14_15_all.py"):
                run(sc)
    else:
        for sc in ("scripts/day7_8_9_all.py", "scripts/day10_11_12_all.py", "scripts/day13_14_15_all.py"):
            run(sc)
# ============================================================
# PART C: label helpers
# ============================================================
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

os.makedirs("data/labels", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)
need = ["data/interim/cleaned_reprojected.tif", "data/interim/smoothed_reflectance.npy", "data/interim/wavelengths_clean.npy"]
for p in need:
    if not os.path.exists(p):
        raise SystemExit(f"{p} not found. Pipeline did not finish; set RUN_PIPELINE = True.")
with rasterio.open(need[0]) as s_:
    tf, crs = s_.transform, s_.crs
refl = np.load(need[1]).astype("float32")
wl = np.load(need[2])
B, H, W = refl.shape
valid = ~np.isnan(refl).any(axis=0)
print(f"\n===== PART C: label helpers | grid {H}x{W} | valid pixels {valid.mean() * 100:.1f}% | CRS {crs} =====")


def bnd(nm):
    return int(np.argmin(np.abs(wl - nm)))


def nd(a, b):
    return (a - b) / (a + b + 1e-8)


def to_lonlat(r, c):
    x, y = tf * (np.asarray(c) + 0.5, np.asarray(r) + 0.5)
    if crs is not None and crs.to_string() != "EPSG:4326":
        from rasterio.warp import transform as wtf
        x, y = wtf(crs, "EPSG:4326", list(np.atleast_1d(x)), list(np.atleast_1d(y)))
    return np.asarray(x), np.asarray(y)


def stretch(a):
    lo, hi = np.nanpercentile(a, (2, 98))
    return np.clip(np.nan_to_num((a - lo) / (hi - lo + 1e-9)), 0, 1)


# --- vegetation score (NDVI + PRI) ---
ndvi = nd(refl[bnd(800)], refl[bnd(670)])
pri = nd(refl[bnd(531)], refl[bnd(570)])
veg = valid & (ndvi > 0.3)
if veg.sum() < 500:
    print("WARNING: very little vegetation in this crop (NDVI>0.3). Using all valid pixels instead.")
    veg = valid.copy()


def z(a):
    m, s_ = np.nanmean(a[veg]), np.nanstd(a[veg]) + 1e-9
    return (a - m) / s_


score = (z(ndvi) + z(pri)) / 2
stress = veg & (score < STRESS_Z)
healthy = veg & (score > 0)
print(f"Vegetation pixels: {veg.sum()} | stress candidates: {stress.sum()} ({stress.sum() / max(veg.sum(), 1) * 100:.1f}%) | healthy refs: {healthy.sum()}")
# --- unsupervised zones ---
rng = np.random.default_rng(42)
Xv = refl[:, valid].T
samp = Xv[rng.choice(len(Xv), size=min(20000, len(Xv)), replace=False)]
pca = PCA(n_components=min(10, B)).fit(samp)
km = KMeans(n_clusters=K_ZONES, n_init=10, random_state=42).fit(pca.transform(samp))
zones = np.full((H, W), -1, dtype="int16")
zones[valid] = km.predict(pca.transform(Xv))
del Xv
# --- field visit candidates ---
def pick(mask, n, min_dist=8):
    rr, cc = np.where(mask)
    chosen = []
    for k in rng.permutation(len(rr)):
        r, c = int(rr[k]), int(cc[k])
        if all((r - a) ** 2 + (c - b) ** 2 >= min_dist ** 2 for a, b in chosen):
            chosen.append((r, c))
            if len(chosen) >= n:
                break
    return chosen


nh = N_CANDIDATES // 2
cand = [("stress_candidate", r, c) for r, c in pick(stress, nh)] + [("healthy_reference", r, c) for r, c in pick(healthy, N_CANDIDATES - nh)]
rows = []
for i, (t, r, c) in enumerate(cand, 1):
    lon, lat = to_lonlat(r, c)
    rows.append({"id": i, "lat": round(float(lat), 6), "lon": round(float(lon), 6), "type": t,
                 "veg_score_z": round(float(score[r, c]), 2), "ndvi": round(float(ndvi[r, c]), 3),
                 "zone": int(zones[r, c]), "row": r, "col": c,
                 "note": "VERIFY IN FIELD - screening point, NOT a disease label"})
cand_df = pd.DataFrame(rows)
cand_df.to_csv("data/labels/field_visit_candidates.csv", index=False)
with open("data/labels/field_visit_candidates.kml", "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n')
    for r in rows:
        f.write(f'<Placemark><name>{r["id"]}_{r["type"]}</name><description>NDVI {r["ndvi"]}, z {r["veg_score_z"]}. Verify in field.</description>'
                f'<Point><coordinates>{r["lon"]},{r["lat"]},0</coordinates></Point></Placemark>\n')
    f.write("</Document></kml>\n")
print(f"Saved {len(rows)} candidate points -> data/labels/field_visit_candidates.csv and .kml (open the .kml in Google Earth)")

rgb = np.dstack([stretch(refl[bnd(650)]), stretch(refl[bnd(550)]), stretch(refl[bnd(470)])])
fig, ax = plt.subplots(1, 3, figsize=(17, 6))
ax[0].imshow(rgb); ax[0].set_title("RGB")
ax[1].imshow(np.ma.masked_equal(zones, -1), cmap="tab10"); ax[1].set_title(f"Unsupervised spectral zones (k={K_ZONES}) - NOT disease classes")
ax[2].imshow(rgb); ax[2].imshow(np.ma.masked_where(~stress, stress), cmap="autumn", alpha=0.6)
for t, col in (("stress_candidate", "red"), ("healthy_reference", "cyan")):
    d = cand_df[cand_df["type"] == t]
    ax[2].scatter(d["col"], d["row"], s=25, c=col, edgecolors="k", label=t)
ax[2].legend(fontsize=7); ax[2].set_title("Stress candidates (orange) + field-visit points")
for a in ax:
    a.axis("off")
plt.tight_layout(); plt.savefig("reports/figures/d22_zones_and_stress.png", dpi=150); plt.close()