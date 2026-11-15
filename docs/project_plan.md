# TerraSpectra: Project Plan (Nov 10 to Dec 10)

**Project:** Hyperspectral Crop Disease Forecasting. A 3D-CNN + Vision Transformer model reads hyperspectral satellite data and predicts a fungal blight outbreak about three weeks before any visible symptoms. A map dashboard shows the predicted risk zones.

## Roles
The table below was written on Nov 10. In practice Kirat took the model work and Kailash the data work, so read the Data and Model columns in the day tables with those two names swapped. Sharath is frontend throughout.


| Member | Role | Owns |
|---|---|---|
| Member 1 (Kirat) | Model and API (as it turned out) | 3D-CNN, patch training, evaluation, prediction tools, later the API |
| Member 3 (Kailash) | Data (as it turned out) | Data sources, cleaning, normalisation, labels, patches, data documents |
| Member 2 (Sharath) | Frontend and GIS | React, Deck.gl map, heatmaps, analytics sidebar, timeline slider |

Folders: `data/` (pipeline code and generated data), `backend/` (model and API), `frontend/`, `docs/`.

## Deliverables (from the assignment letter)
Source code, project documentation, project report, presentation (PPT), model evaluation results, final demonstration.

## Shared formats (agree these first)

**Simulated scene.** Real Hyperion or Sentinel files are gigabytes, so the data is simulated. Healthy plants get a normal vegetation spectrum. Stressed plants get a small shift at the red edge (about 700 to 750 nm), less chlorophyll absorption and a lower near-infrared plateau. Because the simulation knows the truth, we can measure accuracy exactly. Everything must be labelled "simulated data" in the docs and the dashboard.
- Farm cube: 200 bands (400 to 2400 nm), 135 x 135 pixels at 15 m = about 1,013 acres.
- Large scene for tiling and memory tests: 200 x 1024 x 1024 float32 = about 0.84 GB per date.
- 6 acquisition dates, one per week. A 5-acre blight zone starts small and grows.
- Format: GeoTIFF or HDF5, with a coordinate system (EPSG:4326) and GPS transform. After PCA, 20 components.
- Labels per pixel: `stressed` (0/1), `days_to_symptom`.

**Risk map JSON (the API output the frontend reads):**
```json
{
  "date": "2026-11-24",
  "crs": "EPSG:4326",
  "origin": [77.5946, 12.9716],
  "cell_size_deg": 0.000135,
  "rows": 135, "cols": 135,
  "risk": [[0.02, 0.03, "..."]],
  "acres_at_risk": 41.3,
  "anomalies": [{"band_nm": 720, "name": "red edge shift", "score": 0.81}]
}
```
Endpoints: `GET /api/health`, `GET /api/risk-map?date=`, `GET /api/dates`, `POST /api/predict-tile`.

## Day-by-day plan

Each row is a day with at least one real commit per role. "(opt)" means optional if time is short.

### Week 1 (Nov 10 to 16): frontend sprint first, then data and model start
The first three days go to the frontend so there is a working map early. It runs on a mock risk file that follows the shared format, so it needs nothing from the other roles. Data and model work starts on Nov 14.

| Date | Data (Kirat) | Model and API (Kailash) | Frontend (Sharath) |
|---|---|---|---|
| Tue Nov 10 | Plan and data contract (`docs/project_plan.md`) | | |
| Wed Nov 11 | | | Vite + React app, map with Deck.gl, simulated satellite image layer |
| Thu Nov 12 | | | Mock risk JSON (135 x 135 with a growing outbreak zone), risk grid layer, legend, hover tooltip |
| Fri Nov 13 | | | Sidebar (acres at risk, anomaly chart), timeline slider on mock dates, phone layout |
| Sat Nov 14 | Spectral signature simulator (healthy, early stress, visible stress) | `requirements.txt`, project skeleton, config | Frontend README, screenshots, small fixes |
| Sun Nov 15 | Generate the 135 x 135 x 200 cube, write GeoTIFF with Rasterio and GPS transform | PyTorch `Dataset` class reading patches | |
| Mon Nov 16 | HDF5 export, metadata, 6 dates with growing outbreak | NDVI and RGB baseline detector (shows why early detection needs more bands) | |

### Week 2 (Nov 17 to 23): 3D-CNN and mid-review
| Date | Data (Kirat) | Model and API (Kailash) | Frontend (Sharath) |
|---|---|---|---|
| Tue Nov 17 | PCA to 20 components, variance plot | Patch extraction (9 x 9 x 20), split by region to avoid leakage | GPS alignment check against known corner coordinates |
| Wed Nov 18 | Pixel labels (`stressed`, `days_to_symptom`) | Loader tests, hybrid 3D-CNN architecture (Conv3d blocks), shape tests | 5-acre red zone highlight, opacity slider |
| Thu Nov 19 | Normalisation with per-band stats from the training split | Training loop, checkpoints, CPU-friendly settings | Replace the mock file with the real risk output |
| Fri Nov 20 | Large 1024 x 1024 scene generator | First training run, loss curve | Mid-review demo flow |
| Sat Nov 21 | Memory audit script: peak RAM against batch size, chunked reading | Evaluation: accuracy, F1, confusion matrix, comparison with the NDVI baseline | Buffer and fixes |
| Sun Nov 22 | Audit results and charts in `docs/` | Lead-time analysis: how many days before symptoms the model detects stress | Viva notes for the frontend |
| Mon Nov 23 | Tests, run instructions | Save best model, `docs/week2_model_report.md` | Buffer |

Mid-review is probably right after Week 2. Check the portal calendar for the exact date.

### Week 3 (Nov 24 to 30): Vision Transformer and analytics panel
| Date | Data (Kirat) | Model and API (Kailash) | Frontend (Sharath) |
|---|---|---|---|
| Tue Nov 24 | Healthy vs stressed spectral difference per band | Vision Transformer design: group bands into tokens, shape tests | Wire the sidebar to the real anomaly export |
| Wed Nov 25 | Anomaly table export (red edge, chlorophyll, water bands) | Hybrid CNN to ViT model, forward pass | Acres at risk from the real risk map |
| Thu Nov 26 | Pixel to acre conversion with tests | Train the hybrid model | Anomaly chart with the real band scores |
| Fri Nov 27 | Data card (`docs/data_card.md`) | Ablation: NDVI vs 3D-CNN vs hybrid | Zone list; clicking a zone flies the map to it |
| Sat Nov 28 | Augmentation (noise, brightness) option | Tuning; attention weights plot (opt) | Risk threshold slider |
| Sun Nov 29 | Tests for new data code | `docs/week3_report.md` with the comparison table | Loading, error and empty states |
| Mon Nov 30 | Buffer | Buffer | Week 3 write-up draft |

### Week 4 (Dec 1 to 7): API, tiling and timeline
| Date | Data (Kirat) | Model and API (Kailash) | Frontend (Sharath) |
|---|---|---|---|
| Tue Dec 1 | Tiling: slice large scene into overlapping windows | FastAPI skeleton: `/api/health`, `/api/dates` | Timeline slider on the 6 real dates |
| Wed Dec 2 | Stitching with overlap blending, test that tiled result matches the full-scene result | `POST /api/predict-tile` using the tiler | Slider switches the risk layer, smooth transitions |
| Thu Dec 3 | Tiling benchmark: time and memory | `GET /api/risk-map?date=` | Connect to the live API with fallback to bundled JSON |
| Fri Dec 4 | Command line: `python data/run_pipeline.py` | API tests and error handling | Play button animation |
| Sat Dec 5 | Pipeline docs | Dockerfile (note if untested) | Plain-English summary ("this zone is predicted at risk in about 3 weeks") |
| Sun Dec 6 | Integration day: raw cube to API to map, fix what breaks | same | same |
| Mon Dec 7 | Week 4 docs | Week 4 docs | Week 4 write-up |

### Final review window (Dec 8 to 10)
| Date | Everyone |
|---|---|
| Tue Dec 8 | Polish, demo script, rehearse the demo |
| Wed Dec 9 | Project report and PPT, model evaluation results, viva notes |
| Thu Dec 10 | Final submission, last commit, merge to `main` |

## Rules for staying on schedule
- Every day each role should have a real commit (more than 50 changed characters, no duplicate or filler commits).
- If a role falls behind its row, the missing files are prepared here and committed from the Member-2 branch. Work committed on someone's behalf is said openly in the commit message and the team chat.
- Whoever commits a file must be able to explain it in the viva.
- Real data is out of scope. Say so wherever results are shown.

## Status on Nov 15
| Area | Done | Open |
|---|---|---|
| Frontend | Map, risk heatmap, zones, timeline with play, trend and signal charts, phone layout, screenshots | Wire to real model output, 5-acre highlight, risk threshold slider, loading and error states |
| Model | 3D-CNN, patch-based training, checkpoints, early stopping, five-class evaluation, prediction history and summaries | Train on real data, Vision Transformer comparison, ablation, API |
| Data | Data collection plan and preprocessing scripts (Kailash, branch `Member-3`) | Real dataset, label alignment, patch export, merge to `main` |
| Glue | `ml/src/risk_map.py` (model output to dashboard files), `data/make_scene.py` (simulated scans), `scripts/demo_end_to_end.py` | Run with the trained CNN, replace the trend rule with a learned forecaster |
