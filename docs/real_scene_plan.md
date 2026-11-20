# Real EMIT scene in the dashboard (Day 19)

Kailash's `scripts/day21_real_emit_scene.py` (branch Member-3) cuts a real NASA EMIT reflectance crop (285 bands, 381-2493 nm, 60 m).
That scene has **no disease labels**, so the CNN cannot be trained or scored on it. What we can honestly show is a **relative vegetation stress index**.

## What was added
- `scripts/real_scene_to_dashboard.py`: reads his four files in `data/interim/`, drops bad bands, finds vegetation (NDVI >= 0.35), and scores each pixel by how much lower than the scene's typical vegetation its NDVI, NDII (leaf water) and red edge position are (median/MAD z-score, averaged, 3x3 smoothed, 0..1). Writes `frontend/public/data_real/`.
- `scripts/make_fake_emit.py`: a **simulated stand-in** in the same file format so everything runs without the NASA download. It is labelled "Stand-in for EMIT (simulated)" in the dashboard.
- `ml/src/risk_map.py`: `build_payload` now accepts `crop_mask` and an empty field list.
- Dashboard: open with `?data=real`. Titles, legend, badge and panels switch to "stress index"; single-scan scenes get a simple date bar.
- `ml/src/test_real_scene.py`: 5 tests (patch found, soil/water/cloud excluded, bad bands dropped, output contract, too little vegetation).

## How to run on the real scene
    python scripts/real_scene_to_dashboard.py --scene data/interim --out frontend/public/data_real
    cd frontend && npm run dev     # open http://localhost:5173/?data=real

## What the map does NOT mean
High values mean "unusual for this scene". Crop type, growth stage and soil moisture also move these indices, so it is not a disease probability. Real labels (`data/labels/field_points.csv`) are needed before any disease claim.

## Open issues
- `data/interim/*.npy` are large: add `data/interim/` to `.gitignore` (his script runs `git add -A`).
- CNN still misses the visible-stress class; its probabilities are overconfident; forecast is a trend rule.
- Only the first 3 simulated scans have 21-day targets; a longer simulated season is needed.
- No API/Docker yet.
