"""Check that the hyperspectral cube, the field map and the dashboard all use the same grid.

If the cube's GPS transform were off by even a few pixels, the risk heatmap would sit on the wrong fields. This script
compares, for a GeoTIFF scene:
    1. the corner coordinates of the cube with the bounds in farm_meta.json (what the dashboard uses)
    2. the pixel size in metres (should be 15 m)
    3. the corners of every field polygon: each must land on a whole pixel of the cube
and prints a table. Exit code 1 if anything is off by more than the tolerance.

    python data/check_alignment.py data/generated/scene_2026-11-10.tif
"""
import argparse
import json
import math
import sys
from pathlib import Path

TOL_DEG = 1e-6
TOL_PIXEL = 0.05           # a field corner may sit this far (in pixels) from a pixel edge


def metres_per_degree(lat):
    return 111_320.0 * math.cos(math.radians(lat)), 111_320.0


def lonlat_to_pixel(lon, lat, bounds, rows, cols):
    west, south, east, north = bounds
    return (lon - west) / (east - west) * cols, (north - lat) / (north - south) * rows      # column, row


def check(tif_path, farm_dir):
    import rasterio

    farm_dir = Path(farm_dir)
    meta = json.loads((farm_dir / "farm_meta.json").read_text())
    geo = json.loads((farm_dir / "farm_fields.geojson").read_text())
    rows_out = []
    with rasterio.open(tif_path) as src:
        b = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
        rows, cols = src.height, src.width
        crs = src.crs.to_string()
    rows_out.append(("CRS", crs, "EPSG:4326", crs == "EPSG:4326"))
    for name, got, want in zip(("west", "south", "east", "north"), b, meta["bounds"]):
        rows_out.append((f"{name} edge", f"{got:.6f}", f"{want:.6f}", abs(got - want) <= TOL_DEG))
    m_lon, m_lat = metres_per_degree((b[1] + b[3]) / 2)
    px_x, px_y = (b[2] - b[0]) / cols * m_lon, (b[3] - b[1]) / rows * m_lat
    want_px = meta["pixel_size_m"]
    rows_out.append(("pixel width (m)", f"{px_x:.2f}", f"{want_px:.2f}", abs(px_x - want_px) < 0.2))
    rows_out.append(("pixel height (m)", f"{px_y:.2f}", f"{want_px:.2f}", abs(px_y - want_px) < 0.2))
    rows_out.append(("grid size", f"{rows} x {cols}", f"{meta['grid'][0]} x {meta['grid'][1]}", [rows, cols] == meta["grid"]))
    worst = 0.0
    for f in geo["features"]:
        for lon, lat in f["geometry"]["coordinates"][0]:
            c, r = lonlat_to_pixel(lon, lat, b, rows, cols)
            worst = max(worst, abs(c - round(c)), abs(r - round(r)))
    rows_out.append(("field corners vs pixel edges (px)", f"{worst:.4f}", f"<= {TOL_PIXEL}", worst <= TOL_PIXEL))
    return rows_out


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tif")
    ap.add_argument("--farm", default=str(root / "frontend" / "public" / "data"))
    a = ap.parse_args()
    rows = check(a.tif, a.farm)
    for name, got, want, ok in rows:
        print(f"{'OK ' if ok else 'BAD'}  {name:36s} {got:>16s}   expected {want}")
    sys.exit(0 if all(r[3] for r in rows) else 1)


if __name__ == "__main__":
    main()
