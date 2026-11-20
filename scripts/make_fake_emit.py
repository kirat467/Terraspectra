"""Make a FAKE stand-in for Kailash's real EMIT crop, in exactly the file format his script writes.

Why: the real scene needs a NASA Earthdata login and a multi-GB download, so it cannot be fetched everywhere. This stand-in
lets the adapter (scripts/real_scene_to_dashboard.py) and its tests run anywhere. It is SIMULATED and marked as such in the
metadata ("simulated_standin": true). Never present its output as real.

Files written to --out (same names as scripts/day21_real_emit_scene.py on branch Member-3):
    real_scene_cube.npy   float32 (bands, height, width), NaN where masked (cloud) or no data
    real_scene_wl.npy     wavelengths in nm (285 bands, 381 to 2493 nm like EMIT)
    real_scene_good.npy   bool per band, False in the strong water absorption windows
    real_scene_meta.json  granule, scene_date, transform (affine a,b,c,d,e,f), shape_BHW, bbox_lonlat

    python scripts/make_fake_emit.py --out /tmp/fake_emit
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from spectral_simulator import leaf_spectrum, soil_spectrum, water_spectrum  # noqa: E402


def make_fake(size=192, seed=3, lon0=84.9, lat0=25.85, res_x=0.000598, res_y=0.000539, date="20250211"):
    rng = np.random.default_rng(seed)
    wl = np.linspace(381.0, 2493.0, 285)
    good = ~(((wl > 1340) & (wl < 1470)) | ((wl > 1790) & (wl < 1990)) | (wl > 2450))
    yy, xx = np.mgrid[0:size, 0:size]
    sev = np.zeros((size, size))
    nir = np.ones((size, size))
    kind = np.zeros((size, size), dtype=int)                        # 0 crop, 1 bare soil, 2 water
    half = size // 2
    for (r0, r1, c0, c1), scale in [((0, half, 0, half), 1.0), ((0, half, half, size), 1.1), ((half, size, 0, half), 0.92), ((half, size, half, size), 1.04)]:
        nir[r0:r1, c0:c1] = scale * (1 + 0.02 * rng.normal())
    kind[(yy > 0.55 * size) & (yy < 0.60 * size)] = 2                # a river
    kind[(xx < 0.08 * size)] = 1                                     # a bare strip
    d2 = ((xx - 0.72 * size) ** 2 + (yy - 0.28 * size) ** 2) / (0.12 * size) ** 2
    sev = 0.7 * np.exp(-d2)                                          # one stressed patch in the north-east field
    bright = 1 + 0.04 * rng.normal(size=(size, size))
    px = leaf_spectrum(sev.ravel(), nir.ravel(), bright.ravel(), wavelengths=wl)
    cube = px.reshape(size, size, -1)
    cube[kind == 1] = soil_spectrum(wavelengths=wl)[None, :] * bright[kind == 1][:, None]
    cube[kind == 2] = water_spectrum(wavelengths=wl)[None, :]
    cube = cube + 0.004 * rng.normal(size=cube.shape)
    cloud = ((xx - 0.25 * size) ** 2 + (yy - 0.2 * size) ** 2) < (0.07 * size) ** 2
    cube[cloud] = np.nan                                            # masked cloud
    cube[:, :3] = np.nan                                            # no-data edge
    cube = np.ascontiguousarray(np.clip(cube, 0, 1).astype(np.float32).transpose(2, 0, 1))
    west, north = lon0, lat0
    bbox = [west, north + size * res_y, west + size * res_x, north]
    meta = {"granule": "FAKE_EMIT_STANDIN", "scene_date": date, "simulated_standin": True, "crs": "EPSG:4326",
            "transform": [res_x, 0.0, west, 0.0, res_y, north], "shape_BHW": list(cube.shape), "bbox_lonlat": bbox, "crop_window_rc": [0, size, 0, size]}
    return cube, wl.astype(np.float32), good, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=192)
    a = ap.parse_args()
    cube, wl, good, meta = make_fake(a.size)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "real_scene_cube.npy", cube)
    np.save(out / "real_scene_wl.npy", wl)
    np.save(out / "real_scene_good.npy", good)
    (out / "real_scene_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote FAKE EMIT stand-in to", out, cube.shape)


if __name__ == "__main__":
    main()
