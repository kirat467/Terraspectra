# How to run TerraSpectra from scratch

Run every command from the repository root. Python 3.10+ and Node 20+ are needed. Tested on Linux; on Windows use the same commands in PowerShell.

## 1. Install
    pip install -r data/requirements.txt          # numpy, matplotlib, rasterio, h5py, pytest
    pip install -r ml/requirements.txt            # torch and the rest of the model code (this file is UTF-16; if pip complains, install torch scikit-learn scipy pandas by hand)
    cd frontend && npm install && cd ..

## 2. Check everything works
    python scripts/run_all_checks.py --fast       # about 15 seconds; drop --fast to include the large-scene tests and the frontend build

## 3. See the dashboard with the bundled data
    cd frontend && npm run dev                    # open http://localhost:5173, press "Demo" for the guided tour

## 4. Rebuild the simulated data and the model (a few minutes on a CPU)
    python data/make_scene.py                     # 6 scans of the 135 x 135 x 200 farm -> data/generated/ (git-ignored, see below)
    python data/make_patches.py                   # 9 x 9 x 20 patches, block split, PCA -> data/generated/patches.npz
    python ml/src/train_on_scenes.py              # trains the 3D-CNN -> ml/models/cnn_run/, docs/cnn_run_results.json
    python scripts/cnn_to_dashboard.py            # runs the model over the 6 scans -> frontend/public/data/risk_cnn/

## 5. Analyses
    python ml/src/lead_time.py                    # days before symptoms each detector flags stress -> docs/lead_time.json
    python data/make_large_scene.py --size 1024   # 1024 x 1024 x 200 scene, written in strips (419 MB)
    python data/memory_audit.py --cube data/generated/large_cube_5.npy    # peak memory against batch and strip size

## 6. A real EMIT scene
Kailash's `scripts/day21_real_emit_scene.py` (needs NASA Earthdata access) writes four files into `data/interim/`. Then:

    python scripts/real_scene_to_dashboard.py     # -> frontend/public/data_real/
    cd frontend && npm run dev                    # open http://localhost:5173/?data=real

Without the download, `python scripts/make_fake_emit.py --out data/interim` creates a labelled simulated stand-in in the same format.
The real-scene map is a relative stress index, not a disease probability (there are no labels).

## Common problems
- **The map is empty or shows "farm files could not be loaded":** run `python frontend/scripts/make_mock_scene.py` and `python frontend/scripts/make_mock_risk.py`, then reload.
- **`ModuleNotFoundError: rasterio`:** it only matters for GeoTIFF output; `python data/make_scene.py --no-tif` skips it.
- **Out of memory on the large scene:** use `--strip 8` (about 540 MB peak).
- **Generated files are large:** `data/generated/` and `ml/models/` should stay out of git. Add these lines to `.gitignore` if they are missing:

      data/generated/
      ml/models/
