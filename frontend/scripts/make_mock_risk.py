"""Generate SIMULATED disease-risk maps for the dashboard while the real model output is pending.

Each weekly scan gets a risk map on the same 135 x 135 grid as the farm picture. A cell's
risk is the chance (0 to 1) that fungal blight will show up there within the next 21 days.
The outbreak starts small in Field B2 and grows week by week; a second, weaker spot appears
in Field C3 and a faint one in Field A3.

Writes into frontend/public/data/risk/:
    risk_index.json        scan dates and a history of acres at high risk
    risk_<date>.json       one map per scan date (format described in frontend/README.md)

Run from the repository root (after make_mock_scene.py):
    python frontend/scripts/make_mock_risk.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_mock_scene import FIELDS, GRID, OUT, PIXEL_M, SCAN_DATES, SQ_M_PER_ACRE, smooth_noise  # noqa: E402

RISK_DIR = OUT / "risk"
ACRES_PER_PIXEL = PIXEL_M ** 2 / SQ_M_PER_ACRE
HIGH = 0.6          # "high risk" threshold used for acres at risk and zones
MODERATE = 0.3


def field_masks():
    """Which pixels are crops (not tracks), and which field each belongs to."""
    crop = np.zeros((GRID, GRID), dtype=bool)
    owner = np.full((GRID, GRID), -1, dtype=int)
    for k, (_, _, x0, y0, x1, y1) in enumerate(FIELDS):
        crop[y0:y1, x0:x1] = True
        owner[y0:y1, x0:x1] = k
    return crop, owner


def blob(xx, yy, cx, cy, radius, sx=1.0, sy=1.0):
    d2 = ((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2
    return np.exp(-d2 / (radius ** 2))


def risk_map(step, rng_seed, crop, owner):
    """step 0..5 is the scan number; the outbreak grows with it."""
    rng = np.random.default_rng(rng_seed)
    s = step / (len(SCAN_DATES) - 1)
    yy, xx = np.mgrid[0:GRID, 0:GRID].astype(float)
    # warp the coordinates a little so the outbreak is not a perfect circle
    warp_x, warp_y = 3.5 * smooth_noise(GRID, 6, np.random.default_rng(11)), 3.5 * smooth_noise(GRID, 6, np.random.default_rng(12))
    wx, wy = xx + warp_x, yy + warp_y

    base = 0.05 + 0.03 * smooth_noise(GRID, 10, np.random.default_rng(13))
    risk = base.copy()

    amp = 0.18 + 0.80 * s ** 1.4                      # main outbreak, Field B2
    risk += amp * blob(wx, wy, 58, 56, 3 + 8.3 * s ** 1.1, 1.15, 0.9)
    s2 = max(0.0, (step - 1) / 4)                     # second spot, Field C3, from the third scan on
    risk += 0.72 * s2 ** 1.2 * blob(wx, wy, 112, 98, 2 + 5 * s2, 1.0, 0.9)
    risk += 0.30 * s ** 1.3 * blob(wx, wy, 108, 24, 3 + 9 * s, 1.0, 1.0)   # faint patch in Field A3

    risk += 0.012 * rng.normal(size=risk.shape)
    fallow = np.array([crop_name == "Fallow" for _, crop_name, *_ in FIELDS])
    risk[(owner >= 0) & fallow[np.clip(owner, 0, None)]] *= 0.25      # nothing grows on fallow land
    risk = np.clip(risk, 0, 0.99)
    return risk


def components(mask):
    """Connected groups of True cells (4-neighbours), as lists of (row, col)."""
    seen = np.zeros_like(mask, dtype=bool)
    groups = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, cells = [(r, c)], []
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < GRID and 0 <= nx < GRID and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(cells)
    return groups


def main():
    meta = json.loads((OUT / "farm_meta.json").read_text())
    west, south, east, north = meta["bounds"]
    RISK_DIR.mkdir(parents=True, exist_ok=True)
    crop, owner = field_masks()
    history = []

    for step, date in enumerate(SCAN_DATES):
        risk = risk_map(step, 100 + step, crop, owner)
        shown = np.where(crop, np.round(risk, 2), np.nan)

        zones = []
        for cells in components((risk >= HIGH) & crop):
            if len(cells) < 12:
                continue
            ys, xs = np.array([c[0] for c in cells]), np.array([c[1] for c in cells])
            peak = float(risk[ys, xs].max())
            main_field = FIELDS[int(np.bincount(owner[ys, xs]).argmax())]
            zones.append({
                "field": f"Field {main_field[0]}", "crop": main_field[1],
                "acres": round(len(cells) * ACRES_PER_PIXEL, 1), "peak_risk": round(peak, 2),
                "center": [round(west + (xs.mean() + 0.5) / GRID * (east - west), 6),
                           round(north - (ys.mean() + 0.5) / GRID * (north - south), 6)],
            })
        zones.sort(key=lambda z: -z["acres"])
        for i, z in enumerate(zones, 1):
            z["id"] = f"Z{i}"

        fields = []
        for k, (name, crop_name, x0, y0, x1, y1) in enumerate(FIELDS):
            cell = risk[y0:y1, x0:x1]
            fields.append({"name": f"Field {name}", "mean_risk": round(float(cell.mean()), 3), "peak_risk": round(float(cell.max()), 2),
                           "acres_high": round(float((cell >= HIGH).sum() * ACRES_PER_PIXEL), 1),
                           "acres_moderate": round(float(((cell >= MODERATE) & (cell < HIGH)).sum() * ACRES_PER_PIXEL), 1)})

        acres_high = round(float(((risk >= HIGH) & crop).sum() * ACRES_PER_PIXEL), 1)
        acres_mod = round(float((((risk >= MODERATE) & (risk < HIGH)) & crop).sum() * ACRES_PER_PIXEL), 1)
        payload = {
            "date": date, "simulated": True, "horizon_days": 21, "crs": "EPSG:4326",
            "bounds": meta["bounds"], "rows": GRID, "cols": GRID,
            "summary": {"acres_high": acres_high, "acres_moderate": acres_mod, "peak_risk": round(float(risk[crop].max()), 2),
                        "mean_risk": round(float(risk[crop].mean()), 3), "zones": len(zones)},
            "zones": zones, "fields": fields,
            "risk": [[None if np.isnan(v) else float(v) for v in row] for row in shown],
        }
        (RISK_DIR / f"risk_{date}.json").write_text(json.dumps(payload, separators=(",", ":")))
        history.append({"date": date, "acres_high": acres_high, "acres_moderate": acres_mod, "peak_risk": payload["summary"]["peak_risk"]})
        print(f"{date}: high-risk {acres_high:6.1f} ac, moderate {acres_mod:6.1f} ac, zones {len(zones)}, "
              f"cells >=0.8: {int(((risk >= 0.8) & crop).sum())}")

    (RISK_DIR / "risk_index.json").write_text(json.dumps({"dates": SCAN_DATES, "latest": SCAN_DATES[-1], "history": history,
                                                         "high_threshold": HIGH, "moderate_threshold": MODERATE}, indent=2))


if __name__ == "__main__":
    main()
