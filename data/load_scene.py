"""Read the simulated scenes back from disk (HDF5 or GeoTIFF) as plain NumPy arrays.

    scene = load_scene("data/generated/scenes.h5", "2026-11-10")
    scene["cube"]          float32 reflectance (height, width, bands), the layout ml/src expects
    scene["wavelengths_nm"], scene["bounds"] (west, south, east, north), scene["label"], scene["severity"] (HDF5 only)

Real scenes will arrive as GeoTIFF or .mat; keep this function's output shape and everything downstream stays the same.
"""
from pathlib import Path

import numpy as np


def list_dates(path):
    import h5py

    with h5py.File(path) as h:
        return sorted(h["cube"].keys())


def load_scene(path, date=None):
    path = Path(path)
    if path.suffix in (".h5", ".hdf5"):
        return _load_h5(path, date)
    if path.suffix in (".tif", ".tiff"):
        return _load_tif(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _load_h5(path, date):
    import h5py

    with h5py.File(path) as h:
        dates = sorted(h["cube"].keys())
        if date is None:
            date = dates[-1]
        if date not in dates:
            raise KeyError(f"No scan for {date}. Available: {dates}")
        scale = float(h.attrs["scale_factor"])
        return {
            "date": date, "cube": h[f"cube/{date}"][...].astype(np.float32) / scale,
            "wavelengths_nm": h["wavelengths_nm"][...], "bounds": [float(b) for b in h.attrs["bounds"]],
            "pixel_size_m": float(h.attrs["pixel_size_m"]), "label": h[f"label/{date}"][...], "severity": h[f"severity/{date}"][...],
            "simulated": bool(h.attrs["simulated"]),
        }


def _load_tif(path):
    import rasterio

    with rasterio.open(path) as src:
        tags = src.tags()
        scale = float(tags.get("scale_factor", 10000))
        wl = np.array([float(w) for w in tags["wavelengths_nm"].split(",")]) if "wavelengths_nm" in tags else np.array([float(d.split()[0]) for d in src.descriptions])
        b = src.bounds
        return {"date": tags.get("date"), "cube": (src.read().transpose(1, 2, 0) / scale).astype(np.float32), "wavelengths_nm": wl,
                "bounds": [b.left, b.bottom, b.right, b.top], "pixel_size_m": None, "label": None, "severity": None,
                "simulated": tags.get("simulated") == "true"}
