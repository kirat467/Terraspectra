"""Day 21: real EMIT scene -> orthorectified, officially masked crop for the pipeline."""
import os, sys, glob, json, re, shutil, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (change here) =====================
BBOX = (84.0, 25.0, 86.0, 26.5)          # (west, south, east, north)
DATE_RANGE = ("2024-01-01", "2025-12-31")
CENTER_LAT, CENTER_LON = 25.75, 85.0     # crop centre (falls back to scene centroid if outside)
CROP = 512                               # crop size in pixels (512 ~ 30 km at 60 m)
AUTO_PICK_LOWEST_CLOUD = True
SCENE_INDEX = 0                          # used only if AUTO_PICK_LOWEST_CLOUD = False
MASK_WATER = False                       # True = also mask water pixels
ARCHIVE_OLD = True                       # move Indian Pines outputs to data/archive_previous_run
SAVE_TIF = True                          # also save a GeoTIFF (open in QGIS)
# ==================================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "earthaccess", "xarray", "netCDF4", "affine"])

import numpy as np
import xarray as xr
import earthaccess
from affine import Affine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for d in ("data/raw/emit_real", "data/interim", "data/metadata", "data/labels", "reports/figures"):
    os.makedirs(d, exist_ok=True)
# ---------- 1. search + pick scene ----------
earthaccess.login()
res = earthaccess.search_data(short_name="EMITL2ARFL", bounding_box=BBOX, temporal=DATE_RANGE, count=30)
if not res:
    raise SystemExit("No EMIT scenes found. Widen BBOX / DATE_RANGE (EMIT coverage is sparse).")


def cloud_of(r):
    try:
        return float(r["umm"].get("CloudCover"))
    except Exception:
        return 999.0


print(f"{len(res)} scenes found:")
for i, r in enumerate(res):
    print(f"  [{i}] {r['umm']['GranuleUR']} | cloud: {cloud_of(r) if cloud_of(r) < 999 else 'NA'}")
k = int(np.argmin([cloud_of(r) for r in res])) if AUTO_PICK_LOWEST_CLOUD else SCENE_INDEX
granule = res[k]["umm"]["GranuleUR"]
print("Selected:", granule)
# ---------- 2. download only reflectance + mask ----------
links = [l for l in res[k].data_links() if l.endswith(".nc") and "UNCERT" not in l.upper()]
print("Downloading:", [os.path.basename(l) for l in links], "(can be a few GB)")
earthaccess.download(links, "data/raw/emit_real")
nc = glob.glob("data/raw/emit_real/*.nc")
rfl_files = [f for f in nc if "_RFL_" in os.path.basename(f) and "UNCERT" not in f.upper()]
mask_files = [f for f in nc if "_MASK_" in os.path.basename(f)]
if not rfl_files:
    raise SystemExit("Reflectance file not found in data/raw/emit_real. Download may have failed.")
rfl_file = rfl_files[0]
mask_file = mask_files[0] if mask_files else None
print("Reflectance:", os.path.basename(rfl_file), "| Mask:", os.path.basename(mask_file) if mask_file else "NOT FOUND")
# ---------- 3. open scene + GLT ----------
ds = xr.open_dataset(rfl_file)
loc = xr.open_dataset(rfl_file, group="location")
bp = xr.open_dataset(rfl_file, group="sensor_band_parameters")
wl = bp["wavelengths"].values.astype("float32")
good = bp["good_wavelengths"].values.astype(bool) if "good_wavelengths" in bp else np.ones(len(wl), bool)
gtf = np.array(ds.attrs["geotransform"], dtype=float)      # GDAL order: ulx, xres, xskew, uly, yskew, yres
gx = np.nan_to_num(loc["glt_x"].values, nan=0).astype("int64")
gy = np.nan_to_num(loc["glt_y"].values, nan=0).astype("int64")
valid = (gx > 0) & (gy > 0)                                 # GLT indices are 1-based, 0 = no data
Ho, Wo = gx.shape
print(f"Raw bands: {len(wl)} | ortho grid: {Ho} x {Wo} | valid ortho pixels: {valid.mean() * 100:.1f}%")
# ---------- 4. choose crop window ----------
rr = int(round((CENTER_LAT - gtf[3]) / gtf[5]))
cc = int(round((CENTER_LON - gtf[0]) / gtf[1]))


def window(r, c):
    r0 = int(np.clip(r - CROP // 2, 0, max(Ho - CROP, 0)))
    c0 = int(np.clip(c - CROP // 2, 0, max(Wo - CROP, 0)))
    return r0, min(r0 + CROP, Ho), c0, min(c0 + CROP, Wo)


r0, r1, c0, c1 = window(rr, cc)
if valid[r0:r1, c0:c1].mean() < 0.2:
    vr, vc = np.where(valid)
    r0, r1, c0, c1 = window(int(vr.mean()), int(vc.mean()))
    print("Requested centre is outside the scene -> using scene centroid")
gxw, gyw, vw = gx[r0:r1, c0:c1], gy[r0:r1, c0:c1], valid[r0:r1, c0:c1]
if vw.sum() < 1000:
    raise SystemExit("Too few valid pixels in crop. Try another scene (SCENE_INDEX).")
Hw, Ww = r1 - r0, c1 - c0
# ---------- 5. load only the raw rows needed ----------
ys, xs = gyw[vw] - 1, gxw[vw] - 1
rmin, rmax = int(ys.min()), int(ys.max())
print(f"Loading raw rows {rmin}-{rmax} ({rmax - rmin + 1} rows) ...")
slab = ds["reflectance"][rmin:rmax + 1, :, :].values.astype("float32")

bad_raw = np.zeros(slab.shape[:2], bool)
if mask_file:
    mds = xr.open_dataset(mask_file)
    mk = mds["mask"][rmin:rmax + 1, :, :].values
    try:
        names = [str(n).lower() for n in xr.open_dataset(mask_file, group="sensor_band_parameters")["mask_bands"].values]
    except Exception:
        names = []
    use = [i for i, n in enumerate(names)
           if ("cloud" in n or "cirrus" in n or "spacecraft" in n or (MASK_WATER and "water" in n))]
    if not use:
        use = [0, 1, 3, 4]
        print("Mask band names not readable -> using indices 0,1,3,4 (cloud, cirrus, spacecraft, dilated cloud)")
    else:
        print("Mask flags used:", [names[i] for i in use])
    for i in use:
        bad_raw |= np.nan_to_num(mk[..., i]) > 0
else:
    print("WARNING: no mask file -> clouds NOT masked")
# ---------- 6. orthorectify (raw -> map grid via GLT) ----------
yy, xx = ys - rmin, xs
vals = slab[yy, xx, :]
badp = bad_raw[yy, xx]
vals[badp] = np.nan
vals[vals <= -9990] = np.nan
out = np.full((Hw, Ww, slab.shape[2]), np.nan, dtype="float32")
out[vw] = vals
cube = np.ascontiguousarray(out.transpose(2, 0, 1))      # (B, H, W)
del slab, out, vals

mid = cube.shape[0] // 2
print(f"Crop shape (B,H,W): {cube.shape} | masked (cloud/cirrus/etc.): {badp.mean() * 100:.1f}% of valid | "
      f"usable pixels: {np.isfinite(cube[mid]).mean() * 100:.1f}%")

x0 = gtf[0] + c0 * gtf[1] + r0 * gtf[2]
y0 = gtf[3] + c0 * gtf[4] + r0 * gtf[5]
tf = Affine(gtf[1], gtf[2], x0, gtf[4], gtf[5], y0)
lon_min, lon_max = x0, x0 + Ww * gtf[1]
lat_max, lat_min = y0, y0 + Hw * gtf[5]
# ---------- 7. archive old Indian Pines outputs ----------
if ARCHIVE_OLD and not os.path.exists("data/interim/real_scene_cube.npy"):
    arch = "data/archive_previous_run"
    moved = 0
    for sub, keep in (("interim", ()), ("processed", ()), ("metadata", ("weather_daily.csv",))):
        os.makedirs(f"{arch}/{sub}", exist_ok=True)
        for p in glob.glob(f"data/{sub}/*"):
            b = os.path.basename(p)
            if b == ".gitkeep" or b in keep or b.startswith("real_scene"):
                continue
            shutil.move(p, f"{arch}/{sub}/{b}")
            moved += 1
    print(f"Archived {moved} old files to {arch} (weather_daily.csv kept)")

# ---------- 8. save ----------
m = re.search(r"_(\d{8})T", granule)
scene_date = m.group(1) if m else "unknown"
np.save("data/interim/real_scene_cube.npy", cube)
np.save("data/interim/real_scene_wl.npy", wl)
np.save("data/interim/real_scene_good.npy", good)
meta = {"granule": granule, "scene_date": scene_date, "crs": "EPSG:4326",
        "transform": [tf.a, tf.b, tf.c, tf.d, tf.e, tf.f], "shape_BHW": list(cube.shape),
        "bbox_lonlat": [lon_min, lat_min, lon_max, lat_max], "crop_window_rc": [r0, r1, c0, c1]}
json.dump(meta, open("data/interim/real_scene_meta.json", "w"), indent=2)
json.dump({"dataset": f"EMIT_real_{granule}", "sensor": "EMIT", "shape_BHW": list(cube.shape),
           "wavelength_min_nm": float(wl.min()), "wavelength_max_nm": float(wl.max()),
           "nan_percent": round(float(np.isnan(cube).mean() * 100), 3)},
          open("data/metadata/scene_metadata.json", "w"), indent=2)

if SAVE_TIF:
    import rasterio
    with rasterio.open("data/interim/real_scene_ortho.tif", "w", driver="GTiff", height=Hw, width=Ww,
                       count=cube.shape[0], dtype="float32", crs="EPSG:4326", transform=tf,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(cube)
        for i in range(cube.shape[0]):
            dst.set_band_description(i + 1, f"{wl[i]:.1f}nm")
    print("Saved data/interim/real_scene_ortho.tif")