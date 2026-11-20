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