"""Build the simulated hyperspectral farm: 135 x 135 pixels x 200 bands, one cube per scan date.

It uses the same farm as the dashboard (frontend/public/data/farm_meta.json and farm_fields.geojson), so a map drawn
from these cubes lines up with the picture the dashboard shows. Disease starts as a small early-stress core in
Field B2 and spreads week by week; a second spot grows in Field C3 and a faint one in Field A3.

Writes into --out (default data/generated/, ignored by git because the files are large):
    scene_<date>.tif   GeoTIFF, uint16 reflectance x 10000, 200 bands named by wavelength, EPSG:4326 transform
    scenes.h5          all dates in one HDF5 file: cube/<date>, severity/<date>, label/<date>, plus metadata
    scene_summary.json small file with the dates, bands and class counts (safe to commit)

Pixel labels: 0 healthy, 1 early stress (first spectral change, nothing visible), 2 visible stress. -1 = not a crop.

Run from the repository root (needs numpy, matplotlib, rasterio, h5py):
    python data/make_scene.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_simulator import (EARLY_STRESS, N_BANDS, VISIBLE_STRESS, WAVELENGTHS_NM, leaf_spectrum, soil_spectrum,  # noqa: E402
                                water_spectrum)

SCAN_DATES = ["2026-10-06", "2026-10-13", "2026-10-20", "2026-10-27", "2026-11-03", "2026-11-10"]
SCALE = 10000
# crop canopy brightness in the near infrared (some crops reflect more than others)
CROP_NIR = {"Paddy": 1.00, "Sugarcane": 1.08, "Ragi": 0.94, "Maize": 1.04, "Fallow": 0.0}


def field_owner(geojson, bounds, rows, cols):
    """Index of the field covering each pixel, -1 where there is none (tracks)."""
    from matplotlib.path import Path as MplPath

    west, south, east, north = bounds
    lon = west + (np.arange(cols) + 0.5) / cols * (east - west)
    lat = north - (np.arange(rows) + 0.5) / rows * (north - south)
    xx, yy = np.meshgrid(lon, lat)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    owner = np.full(rows * cols, -1, dtype=int)
    for k, f in enumerate(geojson["features"]):
        owner[MplPath(f["geometry"]["coordinates"][0]).contains_points(pts)] = k
    return owner.reshape(rows, cols)


def blob(xx, yy, cx, cy, radius, sx=1.0, sy=1.0):
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / radius ** 2)


def severity_map(step, rows, cols, seed=21):
    """Disease severity 0..1 at scan number `step` (0 to 5). Same places as the dashboard's risk maps."""
    s = step / 5.0
    yy, xx = np.mgrid[0:rows, 0:cols].astype(float)
    rng = np.random.default_rng(seed)
    warp = lambda: 3.0 * np.convolve(rng.normal(size=rows * cols), np.ones(25) / 25, mode="same").reshape(rows, cols)
    wx, wy = xx + warp(), yy + warp()
    sev = 0.12 + 0.88 * s ** 1.3                                   # already a faint early-stress core on the first scan
    out = sev * blob(wx, wy, 58, 56, 3 + 8.3 * s ** 1.1, 1.15, 0.9)
    s2 = max(0.0, (step - 1) / 4)
    out += 0.80 * s2 ** 1.1 * blob(wx, wy, 112, 98, 2 + 5 * s2, 1.0, 0.9)
    out += 0.35 * s ** 1.3 * blob(wx, wy, 108, 24, 3 + 9 * s, 1.0, 1.0)
    return np.clip(out, 0.0, 1.0)


def build_cube(severity, owner, geojson, rng, noise=0.004):
    """Reflectance cube (rows, cols, bands) float32 from a severity map and the field layout."""
    rows, cols = severity.shape
    crops = [f["properties"]["crop"] for f in geojson["features"]]
    nir_scale = np.ones((rows, cols))
    fallow = np.zeros((rows, cols), dtype=bool)
    for k, crop in enumerate(crops):
        m = owner == k
        nir_scale[m] = CROP_NIR.get(crop, 1.0) * (1 + 0.03 * rng.normal())      # each field is a little different
        fallow[m] = crop == "Fallow"
    brightness = 1 + 0.04 * rng.normal(size=(rows, cols))                       # illumination changes pixel to pixel
    canopy = leaf_spectrum((severity * (owner >= 0)).ravel(), nir_scale.ravel(), brightness.ravel())
    cube = np.tile(soil_spectrum(), (rows * cols, 1)) * brightness.ravel()[:, None]     # tracks: bare soil
    crop_mask = ((owner >= 0) & ~fallow).ravel()
    cube[crop_mask] = canopy[crop_mask]
    fb = ((owner >= 0) & fallow).ravel()                                         # fallow: mostly soil, a little weed cover
    cube[fb] = 0.8 * soil_spectrum() + 0.2 * leaf_spectrum(0.0)
    cube = cube.reshape(rows, cols, N_BANDS)
    # three small ponds
    yy, xx = np.mgrid[0:rows, 0:cols]
    for cx, cy, rx, ry in [(66, 62, 5, 3.5), (20, 66, 3.5, 3), (112, 99, 3, 2.5)]:
        cube[(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) < 1] = water_spectrum()
    cube += noise * rng.normal(size=cube.shape)
    return np.clip(cube, 0, 1).astype(np.float32)


def stress_labels(severity, owner, geojson):
    label = np.zeros(severity.shape, dtype=np.int8)
    label[severity >= EARLY_STRESS] = 1
    label[severity >= VISIBLE_STRESS] = 2
    crops = np.array([f["properties"]["crop"] for f in geojson["features"]])
    label[owner < 0] = -1
    label[(owner >= 0) & (crops[np.clip(owner, 0, None)] == "Fallow")] = -1
    return label


def build_scenes(farm_dir, dates=SCAN_DATES, seed=7):
    farm_dir = Path(farm_dir)
    meta = json.loads((farm_dir / "farm_meta.json").read_text())
    geo = json.loads((farm_dir / "farm_fields.geojson").read_text())
    rows, cols = meta["grid"]
    owner = field_owner(geo, meta["bounds"], rows, cols)
    scenes = []
    for step, date in enumerate(dates):
        rng = np.random.default_rng(seed + step)
        sev = severity_map(step, rows, cols)
        scenes.append({"date": date, "cube": build_cube(sev, owner, geo, rng), "severity": sev.astype(np.float32),
                       "label": stress_labels(sev, owner, geo)})
    return meta, geo, owner, scenes


def write_geotiff(path, cube, bounds, date):
    import rasterio
    from rasterio.transform import from_bounds

    rows, cols, bands = cube.shape
    west, south, east, north = bounds
    data = np.round(cube.transpose(2, 0, 1) * SCALE).astype(np.uint16)
    with rasterio.open(path, "w", driver="GTiff", height=rows, width=cols, count=bands, dtype="uint16", crs="EPSG:4326",
                       transform=from_bounds(west, south, east, north, cols, rows), compress="deflate") as dst:
        dst.write(data)
        for i, wl in enumerate(WAVELENGTHS_NM, 1):
            dst.set_band_description(i, f"{wl:.0f} nm")
        dst.update_tags(date=date, scale_factor=str(SCALE), simulated="true", wavelengths_nm=",".join(f"{w:.1f}" for w in WAVELENGTHS_NM))


def write_hdf5(path, scenes, meta):
    import h5py

    with h5py.File(path, "w") as h:
        h.attrs.update({"simulated": True, "crs": "EPSG:4326", "bounds": meta["bounds"], "pixel_size_m": meta["pixel_size_m"],
                        "scale_factor": SCALE, "label_names": ["healthy", "early_stress", "visible_stress"], "nodata_label": -1})
        h.create_dataset("wavelengths_nm", data=WAVELENGTHS_NM)
        for s in scenes:
            h.create_dataset(f"cube/{s['date']}", data=np.round(s["cube"] * SCALE).astype(np.uint16), compression="gzip", compression_opts=4)
            h.create_dataset(f"severity/{s['date']}", data=s["severity"], compression="gzip")
            h.create_dataset(f"label/{s['date']}", data=s["label"], compression="gzip")


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--farm", default=str(root / "frontend" / "public" / "data"), help="folder with farm_meta.json and farm_fields.geojson")
    ap.add_argument("--out", default=str(root / "data" / "generated"))
    ap.add_argument("--no-tif", action="store_true", help="skip the GeoTIFF files (faster)")
    a = ap.parse_args()
    meta, geo, owner, scenes = build_scenes(a.farm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_tif:
        for s in scenes:
            write_geotiff(out / f"scene_{s['date']}.tif", s["cube"], meta["bounds"], s["date"])
    write_hdf5(out / "scenes.h5", scenes, meta)
    summary = {"simulated": True, "grid": meta["grid"], "bands": N_BANDS, "wavelength_nm": [400, 2400], "pixel_size_m": meta["pixel_size_m"],
               "dates": [s["date"] for s in scenes],
               "pixels_by_label": {s["date"]: {n: int((s["label"] == v).sum()) for n, v in [("healthy", 0), ("early_stress", 1), ("visible_stress", 2)]} for s in scenes}}
    (out / "scene_summary.json").write_text(json.dumps(summary, indent=2))
    for d, c in summary["pixels_by_label"].items():
        print(d, c)


if __name__ == "__main__":
    main()
