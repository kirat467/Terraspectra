"""Show a hyperspectral scene (such as Kailash's real EMIT crop) in the dashboard as a vegetation stress map.

Input  : the four files written by scripts/day21_real_emit_scene.py (branch Member-3), found in data/interim/ by default
            real_scene_cube.npy (bands, H, W) with NaN for masked pixels, real_scene_wl.npy, real_scene_good.npy, real_scene_meta.json
Output : a dashboard data folder (default frontend/public/data_real/) with farm_rgb.png, farm_meta.json, an empty farm_fields.geojson,
         risk/risk_<date>.json and risk/risk_index.json. Open the dashboard with  ?data=real  to see it.

WHAT THE MAP MEANS. There are no disease labels for this scene, so nothing here is a disease probability. For every vegetated pixel
(NDVI >= 0.35) it compares greenness (NDVI), leaf water (NDII) and the red edge position with the rest of the scene's vegetation, takes
the average robust z-score of "lower than typical", and maps it to 0..1 (0 = typical or better, 1 = 4 or more robust standard
deviations worse than typical, after a margin of 1). Different crops, growth stages and soil moisture all change these indices, so a
high value means "unusual compared with this scene, worth a look", not "diseased". Field labels (data/labels/field_points.csv) are
needed before a model can say anything about disease.

    python scripts/real_scene_to_dashboard.py --scene data/interim --out frontend/public/data_real
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml" / "src"))
from risk_map import SQ_M_PER_ACRE, build_payload, write_outputs  # noqa: E402
from spectral_baseline import ndii, ndvi, red_edge_position  # noqa: E402

VEG_NDVI = 0.35
MARGIN = 1.0          # robust standard deviations of normal variation that are ignored
SPAN = 3.0            # further deviations over which the index climbs from 0 to 1


def load_scene(folder):
    folder = Path(folder)
    cube = np.load(folder / "real_scene_cube.npy")
    wl = np.load(folder / "real_scene_wl.npy").astype(np.float64)
    good = np.load(folder / "real_scene_good.npy").astype(bool)
    meta = json.loads((folder / "real_scene_meta.json").read_text())
    if cube.ndim != 3 or cube.shape[0] != len(wl):
        raise ValueError("Cube must be (bands, height, width) and match the wavelength list")
    return cube, wl, good, meta


def prepare(cube, wl, good):
    """(H, W, bands) reflectance in 0..1 with unusable bands removed, plus a mask of usable pixels."""
    refl = np.ascontiguousarray(cube.transpose(1, 2, 0)).astype(np.float32)
    refl[refl <= -9990] = np.nan
    keep = good & np.isfinite(refl).any(axis=(0, 1))
    refl, wl = refl[:, :, keep], wl[keep]
    refl = np.clip(refl, 0.0, 1.0)
    usable = np.isfinite(refl).all(axis=2)
    return refl, wl, usable


def robust_z(values, mask):
    """z-scores using the median and 1.4826 * MAD of the masked values."""
    v = values[mask]
    med = np.median(v)
    mad = 1.4826 * np.median(np.abs(v - med)) + 1e-9
    return (values - med) / mad


def stress_map(refl, wl, usable):
    """Return (risk 0..1 with NaN outside vegetation, vegetation mask, the three index maps)."""
    safe = np.nan_to_num(refl, nan=0.0)
    nd, wi, rep = ndvi(safe, wl), ndii(safe, wl), red_edge_position(safe, wl)
    veg = usable & (nd >= VEG_NDVI)
    if veg.sum() < 100:
        raise ValueError("Fewer than 100 vegetated pixels: nothing meaningful to compare")
    z = np.mean([-robust_z(nd, veg), -robust_z(wi, veg), -robust_z(rep, veg)], axis=0)        # higher = worse than typical
    z = np.where(veg, z, np.nan)
    # 3 x 3 average over vegetated neighbours to calm single-pixel noise
    pad = np.pad(z, 1, constant_values=np.nan)
    stack = np.stack([pad[i:i + z.shape[0], j:j + z.shape[1]] for i in range(3) for j in range(3)])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        smooth = np.nanmean(stack, axis=0)
    smooth = np.where(veg, smooth, np.nan)
    risk = np.clip((smooth - MARGIN) / SPAN, 0.0, 1.0)
    return risk, veg, {"ndvi": nd, "ndii": wi, "red_edge_nm": rep}


def pixel_size_m(transform, lat):
    dx = abs(transform[0]) * 111_320.0 * np.cos(np.radians(lat))
    dy = abs(transform[4]) * 111_320.0
    return float(np.sqrt(dx * dy))


def rgb_image(refl, wl):
    from PIL import Image

    def band(nm):
        return refl[:, :, int(np.argmin(np.abs(wl - nm)))]

    chans = []
    for nm in (650, 550, 470):
        b = band(nm)
        lo, hi = np.nanpercentile(b, (2, 98))
        chans.append(np.clip(np.nan_to_num((b - lo) / (hi - lo + 1e-9)), 0, 1))
    rgb = (np.dstack(chans) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def build_dashboard_files(folder, out):
    cube, wl_all, good, meta = load_scene(folder)
    refl, wl, usable = prepare(cube, wl_all, good)
    risk, veg, indices = stress_map(refl, wl, usable)
    h, w = risk.shape
    west, south, east, north = meta["bbox_lonlat"]
    lat = (south + north) / 2
    px = pixel_size_m(meta["transform"], lat)
    standin = bool(meta.get("simulated_standin"))
    date = meta["scene_date"]
    iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) == 8 and date.isdigit() else str(date)
    acres_px = px * px / SQ_M_PER_ACRE
    source = "Stand-in for EMIT (simulated)" if standin else "Real EMIT scene"
    farm_meta = {
        "name": f"EMIT scene, {lat:.2f}°N {((west + east) / 2):.2f}°E", "simulated": standin, "source": source,
        "description": f"{source}: {h} x {w} pixels of {px:.0f} m, {len(wl)} usable bands, scan of {iso}. Shows a relative vegetation stress index, not disease: there are no disease labels for this scene.",
        "center": [(west + east) / 2, lat], "bounds": [west, south, east, north], "grid": [h, w], "pixel_size_m": round(px, 1),
        "farm_acres": round(h * w * acres_px, 1), "bands": int(len(wl)), "wavelength_nm": [float(wl.min()), float(wl.max())], "scan_dates": [iso], "crops": {},
    }
    geo = {"type": "FeatureCollection", "features": []}
    payload = build_payload(np.where(veg, risk, 0.0), iso, farm_meta, geo, horizon_days=0, crop_mask=veg)
    payload.update({"kind": "stress_index", "title": "Vegetation stress index", "legend_title": "Stress index (relative to this scene)",
                    "explain": f"How much lower than typical for this scene the greenness, leaf water and red edge are. Scan of {iso}. Not a disease probability.",
                    "simulated": standin})
    out = Path(out)
    (out / "risk").mkdir(parents=True, exist_ok=True)
    rgb_image(refl, wl).save(out / "farm_rgb.png", optimize=True)
    (out / "farm_meta.json").write_text(json.dumps(farm_meta, indent=2))
    (out / "farm_fields.geojson").write_text(json.dumps(geo))
    write_outputs([payload], out / "risk")
    return {"vegetated_pixels": int(veg.sum()), "usable_pixels": int(usable.sum()), "pixels": int(h * w), "acres_high": payload["summary"]["acres_high"],
            "pixel_size_m": px, "bands_used": int(len(wl)), "source": source}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default=str(ROOT / "data" / "interim"))
    ap.add_argument("--out", default=str(ROOT / "frontend" / "public" / "data_real"))
    a = ap.parse_args()
    info = build_dashboard_files(a.scene, a.out)
    print(json.dumps(info, indent=2))
    print(f"\nOpen the dashboard with ?data=real  (for example http://localhost:5173/?data=real)")


if __name__ == "__main__":
    main()
