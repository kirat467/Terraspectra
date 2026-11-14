"""Turn model predictions on hyperspectral scans into the risk maps the dashboard reads.

Pipeline, one scan at a time:
    cube (H, W, bands) --patch model--> disease probability per pixel --resample--> grid
    grids of the last few scans --trend--> forecast risk for the next `horizon_days`
    forecast grid + field boundaries --> risk_<date>.json   (format: frontend/README.md)

The maths is plain NumPy so it can be tested without a GPU. Any model works as long as it can be wrapped as
`predict_fn(patches) -> probabilities`, where patches has shape (N, 1, bands, size, size), the same
layout the 3D CNN in model_3dcnn.py takes. `torch_predict_fn` does that wrapping for a PyTorch model.

Class 0 is "Healthy" (see label_mapping.py), so disease probability = 1 - P(Healthy).
"""
import json
from pathlib import Path

import numpy as np

HIGH = 0.6          # same thresholds as the dashboard
MODERATE = 0.3
SQ_M_PER_ACRE = 4046.8564


# ---------------------------------------------------------------- 1. one scan -> disease probability per pixel
def disease_probability(probabilities, healthy_index=0):
    """Chance of disease from class probabilities of shape (N, classes)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be 2D (patches, classes)")
    return np.clip(1.0 - probabilities[:, healthy_index], 0.0, 1.0)


def torch_predict_fn(model, device=None, healthy_index=0):
    """Wrap a PyTorch classifier as predict_fn: patches (N, 1, bands, h, w) -> disease probability (N,)."""
    import torch
    import torch.nn.functional as F

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    def predict(patches):
        with torch.inference_mode():
            logits = model(torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32)).to(device))
            return disease_probability(F.softmax(logits, dim=1).cpu().numpy(), healthy_index)

    return predict


def scan_disease_map(cube, predict_fn, patch_size=16, stride=8, batch_size=64):
    """Slide a window over the cube and average the disease probability of every window covering each pixel.

    predict_fn receives (N, 1, bands, patch_size, patch_size) and returns N disease probabilities.
    """
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError("Cube must be (height, width, bands)")
    if not np.isfinite(cube).all():
        raise ValueError("Cube contains invalid values")
    height, width, _ = cube.shape
    if height < patch_size or width < patch_size:
        raise ValueError("Cube is smaller than one patch")

    rows = list(range(0, height - patch_size + 1, stride))
    cols = list(range(0, width - patch_size + 1, stride))
    if rows[-1] != height - patch_size:           # make sure the bottom and right edges are covered
        rows.append(height - patch_size)
    if cols[-1] != width - patch_size:
        cols.append(width - patch_size)
    corners = [(r, c) for r in rows for c in cols]

    total = np.zeros((height, width))
    count = np.zeros((height, width))
    for start in range(0, len(corners), batch_size):
        batch = corners[start:start + batch_size]
        patches = np.stack([cube[r:r + patch_size, c:c + patch_size, :].transpose(2, 0, 1) for r, c in batch])[:, None]
        p = np.asarray(predict_fn(patches), dtype=np.float64).reshape(-1)
        if p.shape[0] != len(batch):
            raise ValueError("predict_fn must return one probability per patch")
        for (r, c), v in zip(batch, p):
            total[r:r + patch_size, c:c + patch_size] += v
            count[r:r + patch_size, c:c + patch_size] += 1
    return total / count


def resample_mean(array, rows, cols):
    """Shrink or stretch a 2D map to rows x cols by area averaging (keeps the average risk)."""
    array = np.asarray(array, dtype=np.float64)
    h, w = array.shape
    if (h, w) == (rows, cols):
        return array.copy()
    ys = np.minimum((np.arange(rows + 1) * h / rows).astype(int), h)
    xs = np.minimum((np.arange(cols + 1) * w / cols).astype(int), w)
    out = np.empty((rows, cols))
    for i in range(rows):
        y0, y1 = ys[i], max(ys[i + 1], ys[i] + 1)
        for j in range(cols):
            x0, x1 = xs[j], max(xs[j + 1], xs[j] + 1)
            out[i, j] = array[y0:y1, x0:x1].mean()
    return out


# ---------------------------------------------------------------- 2. several scans -> forecast
def day_number(iso):
    return np.datetime64(iso, "D").astype(int)


def forecast_risk(maps, dates, horizon_days=21, window=3, damping=0.5):
    """Forecast the chance of disease within `horizon_days` of the latest scan.

    Baseline heuristic, not a learned forecaster: take the latest disease map, add the per-pixel trend over
    the last `window` scans carried forward `horizon_days`, damped by `damping` because outbreaks do not grow
    in a straight line forever. Only worsening trends add risk; a pixel that is improving keeps its current level.
    Replace this function with a trained time-series model when enough labelled scans exist.
    """
    if len(maps) != len(dates) or not maps:
        raise ValueError("Need one map per date")
    order = np.argsort([day_number(d) for d in dates])
    maps = [np.asarray(maps[i], dtype=np.float64) for i in order]
    days = np.array([day_number(dates[i]) for i in order], dtype=np.float64)
    current = maps[-1]
    if len(maps) < 2:
        return np.clip(current, 0, 1)
    recent = np.stack(maps[-window:])
    t = days[-window:] - days[-window:].mean()
    slope = (t[:, None, None] * (recent - recent.mean(axis=0))).sum(axis=0) / (t ** 2).sum()    # per day
    return np.clip(current + damping * np.maximum(slope, 0) * horizon_days, 0.0, 0.99)


# ---------------------------------------------------------------- 3. grid -> dashboard JSON
def field_masks(geojson, bounds, rows, cols):
    """Boolean mask of crop pixels and an owner index (-1 = track) from the field polygons."""
    from matplotlib.path import Path as MplPath

    west, south, east, north = bounds
    lon = west + (np.arange(cols) + 0.5) / cols * (east - west)
    lat = north - (np.arange(rows) + 0.5) / rows * (north - south)
    xx, yy = np.meshgrid(lon, lat)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    owner = np.full(rows * cols, -1, dtype=int)
    for k, feature in enumerate(geojson["features"]):
        ring = feature["geometry"]["coordinates"][0]
        owner[MplPath(ring).contains_points(points)] = k
    owner = owner.reshape(rows, cols)
    return owner >= 0, owner


def components(mask, min_cells=12):
    """Connected groups (4-neighbours) of True cells with at least min_cells cells."""
    rows, cols = mask.shape
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
                if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(cells) >= min_cells:
            groups.append(cells)
    return groups


def build_payload(risk, date, meta, geojson, horizon_days=21):
    """The risk_<date>.json document (see frontend/README.md for the format)."""
    risk = np.asarray(risk, dtype=np.float64)
    rows, cols = risk.shape
    west, south, east, north = meta["bounds"]
    crop, owner = field_masks(geojson, meta["bounds"], rows, cols)
    acres_px = meta["pixel_size_m"] ** 2 / SQ_M_PER_ACRE
    features = geojson["features"]

    zones = []
    for cells in components((risk >= HIGH) & crop):
        ys, xs = np.array([c[0] for c in cells]), np.array([c[1] for c in cells])
        main = features[int(np.bincount(owner[ys, xs]).argmax())]["properties"]
        zones.append({"field": main["name"], "crop": main["crop"], "acres": round(len(cells) * acres_px, 1),
                      "peak_risk": round(float(risk[ys, xs].max()), 2),
                      "center": [round(west + (xs.mean() + 0.5) / cols * (east - west), 6),
                                 round(north - (ys.mean() + 0.5) / rows * (north - south), 6)]})
    zones.sort(key=lambda z: -z["acres"])
    for i, z in enumerate(zones, 1):
        z["id"] = f"Z{i}"

    fields = []
    for k, feature in enumerate(features):
        cell = risk[owner == k]
        fields.append({"name": feature["properties"]["name"],
                       "mean_risk": round(float(cell.mean()), 3) if cell.size else 0.0,
                       "peak_risk": round(float(cell.max()), 2) if cell.size else 0.0,
                       "acres_high": round(float((cell >= HIGH).sum() * acres_px), 1),
                       "acres_moderate": round(float(((cell >= MODERATE) & (cell < HIGH)).sum() * acres_px), 1)})

    shown = np.where(crop, np.round(risk, 2), np.nan)
    return {
        "date": date, "simulated": bool(meta.get("simulated", False)), "horizon_days": horizon_days, "crs": "EPSG:4326",
        "bounds": meta["bounds"], "rows": rows, "cols": cols,
        "summary": {"acres_high": round(float(((risk >= HIGH) & crop).sum() * acres_px), 1),
                    "acres_moderate": round(float((((risk >= MODERATE) & (risk < HIGH)) & crop).sum() * acres_px), 1),
                    "peak_risk": round(float(risk[crop].max()), 2), "mean_risk": round(float(risk[crop].mean()), 3),
                    "zones": len(zones)},
        "zones": zones, "fields": fields,
        "risk": [[None if np.isnan(v) else float(v) for v in row] for row in shown],
    }


def write_outputs(payloads, out_dir):
    """Write one file per scan plus risk_index.json (the dashboard loads every date listed in the index)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = sorted(payloads, key=lambda p: p["date"])
    for p in payloads:
        (out_dir / f"risk_{p['date']}.json").write_text(json.dumps(p, separators=(",", ":")))
    index = {
        "dates": [p["date"] for p in payloads], "latest": payloads[-1]["date"],
        "history": [{"date": p["date"], "acres_high": p["summary"]["acres_high"], "acres_moderate": p["summary"]["acres_moderate"],
                     "peak_risk": p["summary"]["peak_risk"], "mean_risk": p["summary"]["mean_risk"], "zones": p["summary"]["zones"]}
                    for p in payloads],
        "high_threshold": HIGH, "moderate_threshold": MODERATE,
    }
    (out_dir / "risk_index.json").write_text(json.dumps(index, indent=2))
    return index


def run(scans, predict_fn, meta, geojson, out_dir, patch_size=16, stride=8, horizon_days=21, window=3):
    """scans: list of (date 'YYYY-MM-DD', cube). Returns the index written to out_dir."""
    rows, cols = meta["grid"]
    maps, payloads = [], []
    for date, cube in sorted(scans, key=lambda s: s[0]):
        maps.append(resample_mean(scan_disease_map(cube, predict_fn, patch_size, stride), rows, cols))
        risk = forecast_risk(maps, [d for d, _ in sorted(scans, key=lambda s: s[0])][:len(maps)], horizon_days, window)
        payloads.append(build_payload(risk, date, meta, geojson, horizon_days))
    return write_outputs(payloads, out_dir)
