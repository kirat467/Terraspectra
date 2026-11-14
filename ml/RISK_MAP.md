# From model predictions to the dashboard risk map

`ml/src/risk_map.py` connects the trained model to the dashboard in `frontend/`.

```
scan cube (H, W, bands) -> sliding patches -> model -> disease probability per pixel
-> average to the 135 x 135 farm grid -> trend over the last scans -> 21-day risk
-> risk_<date>.json files + risk_index.json   (read by the dashboard)
```

## Use it with the 3D CNN

```python
import json
from risk_map import run, torch_predict_fn
from model_3dcnn import Crop3DCNN

model = Crop3DCNN(num_classes=5)            # load trained weights here
predict = torch_predict_fn(model)           # class 0 = Healthy, so disease = 1 - P(Healthy)

meta = json.load(open("frontend/public/data/farm_meta.json"))
fields = json.load(open("frontend/public/data/farm_fields.geojson"))
scans = [("2026-11-03", cube_nov3), ("2026-11-10", cube_nov10)]      # (date, cube of shape H x W x bands)

run(scans, predict, meta, fields, "frontend/public/data/risk", patch_size=16, stride=8)
```

Reload the dashboard and the timeline, map, zones and charts show the model output.

## What is real and what is a placeholder
- Real and tested (`python -m pytest ml/src/test_risk_map.py`): patch sliding, averaging to the grid, JSON format, zones, fields.
- Placeholder: `forecast_risk` is a simple trend rule (latest map plus the damped growth of the last 3 scans). Replace it with a trained forecaster when labelled time series exist.
- `torch_predict_fn` is written for the 3D CNN input layout (N, 1, bands, h, w) but has not been run against the trained model yet.
- The dashboard's spectral-signal chart only appears when `risk_index.json` has a `signals` entry; model output does not include it.
