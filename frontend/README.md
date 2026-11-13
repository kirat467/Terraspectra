# TerraSpectra Frontend

React + Deck.gl map dashboard. It shows the farm as a satellite-style picture with field boundaries, and later the model's disease-risk heatmap.

All imagery is **simulated** (drawn by `scripts/make_mock_scene.py`) until the real hyperspectral dataset is available. The simulated farm is 135 x 135 pixels at 15 m per pixel, about 1,013 acres.

## Run it

You need Node.js 18 or newer (`node -v` to check).

```bash
cd frontend
npm install
npm run dev          # opens on http://localhost:5173
```

Build a production copy and serve it:

```bash
npm run build
npm run preview      # http://localhost:4173
```

## What is on the page

- Satellite-style farm picture placed at real GPS coordinates (a fictional farm near Mandya, Karnataka).
- Field boundaries and labels. Hover a field for details, click it (or pick it in the list) to zoom in.
- Layer switches, imagery opacity, zoom buttons, "show whole farm", scale bar and a live coordinate readout.
- **Disease-risk heatmap** laid over the farm, with a colour legend. Hover to read the risk of the cell under the pointer.
- **Predicted outbreak risk** panel: acres at high risk, peak risk, and a list of risk zones. Click a zone to fly to it.
- Fields listed highest risk first, each with a risk chip; the selected field shows its mean and peak risk.
- Farm summary and the six simulated weekly scan dates. (A timeline slider will step through them.)

## Files

```text
frontend/
  scripts/make_mock_scene.py     writes the simulated farm into public/data/
  scripts/make_mock_risk.py      writes simulated risk maps into public/data/risk/
  public/data/                   farm_rgb.png, farm_fields.geojson, farm_meta.json
  public/data/risk/              risk_index.json and one risk_<date>.json per scan
  src/App.jsx                    map, layers, selection, camera
  src/components/Sidebar.jsx     farm facts, layer switches, field list
  src/components/MapControls.jsx zoom buttons, scale bar, coordinates, risk under the pointer
  src/components/RiskLegend.jsx  colour scale
  src/lib/geo.js                 camera fitting, scale bar maths, formatting
  src/lib/risk.js                risk colours, labels, heatmap picture, cell lookup
  src/styles.css
```

To regenerate the simulated data (needs numpy and pillow), from the repository root:

```bash
python frontend/scripts/make_mock_scene.py
python frontend/scripts/make_mock_risk.py
```

## Risk map format

The dashboard reads one file per scan date. The real model's API should return the same shape, so the mock files can be swapped out later.

```json
{
  "date": "2026-11-10",
  "horizon_days": 21,
  "crs": "EPSG:4326",
  "bounds": [west, south, east, north],
  "rows": 135, "cols": 135,
  "risk": [[0.05, 0.07, null, "..."]],
  "summary": {"acres_high": 15.4, "acres_moderate": 30.0, "peak_risk": 0.99, "mean_risk": 0.089, "zones": 2},
  "zones": [{"id": "Z1", "field": "Field B2", "crop": "Paddy", "acres": 13.5, "peak_risk": 0.99, "center": [lon, lat]}],
  "fields": [{"name": "Field B2", "mean_risk": 0.2, "peak_risk": 0.99, "acres_high": 13.5, "acres_moderate": 9.0}]
}
```

- `risk` is row by row from the north edge. Each cell is the chance (0 to 1) that blight shows up within `horizon_days`. `null` means no crop there (a farm track).
- High risk is 0.6 or more, moderate is 0.3 to 0.6.
- `risk_index.json` lists the scan dates, the latest one, and a history of acres at high risk.
