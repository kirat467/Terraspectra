"""End-to-end pipeline: raw GeoTIFF -> cleaned -> patches -> spatial split -> save.
Usage: python scripts/run_pipeline.py --config config/config.yaml --scene data/raw/scene.tif [--labels data/labels/labels.tif]
"""
import argparse, sys, json
from pathlib import Path
import numpy as np, yaml, rasterio

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_cube, describe
from src.cleaning import bad_band_indices, remove_bands, clip_reflectance
from src.preprocessing import normalize, smooth_spectra
from src.patches import make_patches
from src.split import spatial_block_split
from src.validate import validate_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/config.yaml")
ap.add_argument("--scene", required=True)
ap.add_argument("--labels", default=None)
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
pp, ds = cfg["preprocessing"], cfg["dataset"]

cube, profile, wl = load_cube(a.scene)
describe(cube, wl)
if wl is None:
    raise SystemExit("Wavelengths nahi mile. data/metadata me wavelength file banao aur load karo.")

cube = clip_reflectance(cube, pp["reflectance_scale"])
bad = bad_band_indices(wl, cfg["sensor"]["bad_band_ranges_nm"])
cube, wl, keep = remove_bands(cube, wl, bad)
cube = smooth_spectra(cube, pp["savgol_window"], pp["savgol_polyorder"])
cube = normalize(cube, pp["normalization"])

labels = None
if a.labels:
    with rasterio.open(a.labels) as s:
        labels = s.read(1).astype("int32")

X, y, coords = make_patches(cube, labels, ds["patch_size"], ds["stride"], ds["max_nan_fraction"])
idx = spatial_block_split(coords, ds["patch_size"], ds["block_size_patches"],
                          tuple(ds["split"].values()), ds["seed"])
validate_dataset(X, y, idx)

out = Path(cfg["paths"]["processed"]); out.mkdir(parents=True, exist_ok=True)
np.save(out / "X.npy", X.astype("float32"))
if y is not None: np.save(out / "y.npy", y)
np.save(out / "coords.npy", coords)
json.dump({k: v.tolist() for k, v in idx.items()}, open(out / "split_indices.json", "w"))
json.dump({"wavelengths_nm": wl.tolist(), "kept_band_indices": keep.tolist(),
           "crs": str(profile.get("crs")), "patch_size": ds["patch_size"], "seed": ds["seed"]},
          open(Path(cfg["paths"]["metadata"]) / "dataset_metadata.json", "w"), indent=2)
print("Saved:", X.shape)
