# Simulated hyperspectral scenes

The real dataset is still pending, so these scripts build a stand-in that the rest of the project can use now.

| File | What it does |
|---|---|
| `spectral_simulator.py` | Reflectance of a crop canopy from 400 to 2400 nm (200 bands) for any disease severity from 0 to 1, plus soil and water. |
| `make_scene.py` | Builds one 135 x 135 x 200 cube per scan date on the same farm as the dashboard. Writes GeoTIFF files and one HDF5 file. |
| `load_scene.py` | Reads a scan back from the HDF5 or GeoTIFF file as a NumPy cube (height, width, bands) with wavelengths, bounds and labels. |
| `check_alignment.py` | Checks that the cube's GPS transform matches the dashboard's farm grid (edges, 15 m pixels, field corners on whole pixels). |
| `make_patches.py` | PCA to 20 bands (fitted on training blocks only) and 9 x 9 x 20 patches with a block-based train/val/test split that cannot leak. |
| `tests/` | `python -m pytest data/tests` |

```bash
pip install -r data/requirements.txt
python data/make_scene.py          # writes data/generated/ (about 70 MB, ignored by git)
python data/check_alignment.py data/generated/scene_2026-11-10.tif
python data/make_patches.py        # writes data/generated/patches.npz and docs/figures/pca_variance.png
```

## What the simulation assumes
- Chlorophyll loss moves the red edge toward blue straight away, but changes visible colour only once the loss is large.
- Leaf water loss makes the water dips shallower and raises shortwave infrared reflectance.
- Leaf structure collapses only late, so the near infrared plateau drops last.
- Result: early stress shows in the red edge and water bands while NDVI barely moves. `ml/src/spectral_baseline.py` measures this.

## Pixel labels
`0` healthy, `1` early stress (first spectral change, nothing visible yet), `2` visible stress, `-1` not a crop (track, pond, fallow).

## Limits
This is a model of the physics, not a measurement. A detector trained and tested on it will look better than on real fields. Use it to build and test the pipeline, and report results on real data separately.
