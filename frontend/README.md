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
- Farm summary and the six simulated weekly scan dates.
- The disease-risk layer is not added yet.

## Files

```text
frontend/
  scripts/make_mock_scene.py     writes the simulated farm into public/data/
  public/data/                   farm_rgb.png, farm_fields.geojson, farm_meta.json
  src/App.jsx                    map, layers, selection, camera
  src/components/Sidebar.jsx     farm facts, layer switches, field list
  src/components/MapControls.jsx zoom buttons, scale bar, coordinates
  src/lib/geo.js                 camera fitting, scale bar maths, formatting
  src/styles.css
```

To regenerate the simulated farm (needs numpy and pillow):

```bash
python frontend/scripts/make_mock_scene.py
```
