"""Generate the SIMULATED farm used by the dashboard while real data is pending.

Writes three files into frontend/public/data/:
    farm_rgb.png        top-down "satellite" picture of the farm (drawn, not a real photo)
    farm_fields.geojson field boundaries with crop and acreage
    farm_meta.json      location, size, scan dates

The farm is 135 x 135 pixels at 15 m per pixel = about 1,013 acres, the same grid the
risk map will use later, so the picture and the risk cells line up.

Run from the repository root:
    python frontend/scripts/make_mock_scene.py        (needs numpy and pillow)
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "public" / "data"
GRID = 135                 # pixels per side of the risk grid
PIXEL_M = 15.0             # metres per pixel
SCALE = 6                  # picture is drawn 6x finer than the grid
CENTER = (76.8951, 12.5218)   # lon, lat: a fictional farm near Mandya, Karnataka
SQ_M_PER_ACRE = 4046.8564
SCAN_DATES = ["2026-10-06", "2026-10-13", "2026-10-20", "2026-10-27", "2026-11-03", "2026-11-10"]

# colour of each crop as seen from above (R, G, B)
CROPS = {
    "Paddy": (92, 150, 58),
    "Sugarcane": (44, 108, 52),
    "Ragi": (128, 156, 64),
    "Maize": (78, 134, 48),
    "Fallow": (168, 138, 100),
}
ROAD = (196, 176, 140)

# Field layout on the 135 x 135 grid: (name, crop, x0, y0, x1, y1), edges exclusive.
# The gaps between fields are farm tracks and canals.
FIELDS = [
    ("A1", "Paddy",     2,   2,  42,  40),
    ("A2", "Sugarcane", 46,  2,  86,  40),
    ("A3", "Maize",     90,  2, 133,  44),
    ("B1", "Ragi",      2,  44,  40,  86),
    ("B2", "Paddy",     44, 44,  86,  80),
    ("B3", "Fallow",    90, 48, 133,  80),
    ("C1", "Maize",     2,  90,  42, 133),
    ("C2", "Sugarcane", 46,  84,  86, 133),
    ("C3", "Paddy",     90, 84, 133, 110),
    ("C4", "Ragi",      90, 114, 133, 133),
]


def smooth_noise(size, sigma, rng):
    """Blurry random texture with values roughly in -1..1."""
    white = rng.normal(size=(size, size))
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    lowpass = np.exp(-2 * (math.pi * sigma) ** 2 * (fx ** 2 + fy ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(white) * lowpass))
    return out / (np.abs(out).max() + 1e-9)


def draw_picture(rng):
    n = GRID * SCALE
    img = np.zeros((n, n, 3), dtype=np.float32)
    img[:] = ROAD
    big, mid, fine = smooth_noise(n, 60, rng), smooth_noise(n, 14, rng), smooth_noise(n, 2.5, rng)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)

    for k, (name, crop, x0, y0, x1, y1) in enumerate(FIELDS):
        sl = (slice(y0 * SCALE, y1 * SCALE), slice(x0 * SCALE, x1 * SCALE))
        angle = (k * 37 % 60 + 10) * math.pi / 180          # crop rows run in a different direction per field
        period = 7 + (k % 3) * 2
        rows = np.sin(2 * math.pi * (xx[sl] * math.cos(angle) + yy[sl] * math.sin(angle)) / period)
        base = np.array(CROPS[crop], dtype=np.float32)
        shade = 1 + 0.16 * big[sl] + 0.10 * mid[sl] + 0.05 * fine[sl]
        stripe = 0.05 if crop != "Fallow" else 0.02
        img[sl] = base * (shade + stripe * rows)[..., None]
        # darker edge where crops meet the track
        edge = np.ones(shade.shape, dtype=np.float32)
        edge[:3] = edge[-3:] = 0.82
        edge[:, :3] = edge[:, -3:] = 0.82
        img[sl] *= edge[..., None]

    # farm ponds
    for cx, cy, rx, ry in [(66, 62, 5, 3.5), (20, 66, 3.5, 3), (112, 99, 3, 2.5)]:
        inside = ((xx / SCALE - cx) / rx) ** 2 + ((yy / SCALE - cy) / ry) ** 2 < 1
        img[inside] = (52, 96, 120)
    # tree clumps along the tracks
    on_track = (img == np.array(ROAD, dtype=np.float32)).all(axis=-1)
    placed = 0
    while placed < 70:
        tx, ty = rng.uniform(0, GRID, 2)
        if not on_track[int(ty * SCALE), int(tx * SCALE)]:      # trees only grow along the tracks
            continue
        r = rng.uniform(0.5, 1.0)
        inside = ((xx / SCALE - tx) ** 2 + (yy / SCALE - ty) ** 2) < r ** 2
        img[inside] = (30, 78, 38)
        placed += 1
    # dust on the tracks
    track = (img == np.array(ROAD, dtype=np.float32)).all(axis=-1)
    img[track] *= (1 + 0.05 * fine[track])[:, None]

    picture = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    return picture


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    draw_picture(rng).save(OUT / "farm_rgb.png", optimize=True)

    lon0, lat0 = CENTER
    dlat = PIXEL_M / 111_320.0
    dlon = PIXEL_M / (111_320.0 * math.cos(math.radians(lat0)))
    west, east = lon0 - GRID / 2 * dlon, lon0 + GRID / 2 * dlon
    north, south = lat0 + GRID / 2 * dlat, lat0 - GRID / 2 * dlat

    def to_lonlat(px, py):                       # pixel column/row -> lon, lat (row 0 is the north edge)
        return [round(west + px * dlon, 6), round(north - py * dlat, 6)]

    acres_per_pixel = PIXEL_M ** 2 / SQ_M_PER_ACRE
    features, total = [], 0.0
    for name, crop, x0, y0, x1, y1 in FIELDS:
        px = (x1 - x0) * (y1 - y0)
        acres = round(px * acres_per_pixel, 1)
        total += acres
        ring = [to_lonlat(x0, y0), to_lonlat(x1, y0), to_lonlat(x1, y1), to_lonlat(x0, y1), to_lonlat(x0, y0)]
        features.append({
            "type": "Feature",
            "properties": {"name": f"Field {name}", "crop": crop, "acres": acres, "pixels": px,
                           "center": to_lonlat((x0 + x1) / 2, (y0 + y1) / 2)},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    (OUT / "farm_fields.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    meta = {
        "name": "Demo Farm, Mandya",
        "simulated": True,
        "center": [lon0, lat0],
        "bounds": [round(west, 6), round(south, 6), round(east, 6), round(north, 6)],   # west, south, east, north
        "grid": [GRID, GRID],
        "pixel_size_m": PIXEL_M,
        "farm_acres": round(GRID * GRID * acres_per_pixel, 1),
        "field_acres": round(total, 1),
        "bands": 200,
        "wavelength_nm": [400, 2400],
        "scan_dates": SCAN_DATES,
        "crops": {k: "#%02x%02x%02x" % v for k, v in CROPS.items()},
    }
    (OUT / "farm_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"farm {meta['farm_acres']} acres, {len(FIELDS)} fields, bounds {meta['bounds']}")


if __name__ == "__main__":
    main()
