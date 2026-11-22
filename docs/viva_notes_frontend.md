# Viva notes: the frontend

Short answers to the questions a reviewer is likely to ask about the dashboard. Code is in `frontend/src/`.

**What is it built with, and why?** React 19 with Vite for the interface, deck.gl for the map. deck.gl draws images and polygons on the GPU, so a 135 x 135 risk grid and field outlines stay smooth while zooming.

**What does the dashboard read?** Only files: `farm_meta.json`, `farm_fields.geojson`, `farm_rgb.png`, and one `risk_<date>.json` per scan plus `risk_index.json` (`ml/RISK_MAP.md` documents the format). That is the contract between the model side and the screen: the model team can change the model without touching the frontend as long as the files keep this shape.

**How is the risk map drawn?** Each risk grid becomes one small image with a colour scale, drawn once per scan; stepping through dates just swaps the image, so it is instant. Nearest-neighbour filtering keeps pixels square instead of blurring them.

**How do the priority zones work?** `lib/zones.js` thresholds the grid at the chosen risk level (default 0.6), groups neighbouring cells into connected regions, drops regions smaller than the chosen area (default 5 acres), and ranks them by size. Area comes from the pixel size (15 m, so 0.056 acres per cell). It is tested in `frontend/scripts/test_zones.mjs`.

**How do you know the map lines up with the ground?** The image and the risk grid share the same bounds, and `data/check_alignment.py` checks the corner coordinates of the cubes against the farm file.

**What does the demo mode do?** Six scripted steps move the map and panels with the same controls a user has (`lib/tour.js`). Nothing is faked; it is the live data.

**What is shown for the real EMIT scene?** `?data=real` shows a relative vegetation stress index (greenness, leaf water, red edge compared with the rest of the scene). It is not a disease probability: that scene has no disease labels.

**What are the honest limits?**
- All farm data is simulated; the dashboard says "Simulated data" in two places.
- The forecast is a damped trend on each pixel's scores, not a learned model.
- The CNN misses most visible-stress pixels (recall 0.28 on the test blocks) and its probabilities are overconfident (`docs/first_training_run.md`).
- Lead time: detectors flag stress about 14 days before symptoms in simulation, which is the head start built into it (`docs/lead_time_analysis.md`).
- There is no live backend yet: the files are produced by scripts and copied into `public/`.

**What would you do next?** Field visits to get real labels, retrain, a small API so a new scan can be uploaded, and test on a phone in sunlight (the layout works at phone width, but contrast outdoors is untested).
