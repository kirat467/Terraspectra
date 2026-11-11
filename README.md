# TerraSpectra: Hyperspectral Crop Disease Forecasting
## Meri responsibility: Data Collection & Preparation (Kailash Kumar)

Company: Infotact Solutions | Ref: ITS/DSML/1513
Project duration: 10/11/2026 - 10/12/2026

## Folder structure
```
config/        -> config.yaml (sab paths/parameters yahin)
data/raw       -> original downloaded data (kabhi edit nahi karna)
data/interim   -> cleaned / reprojected data
data/processed -> final patches, splits (model team ko jaata hai)
data/labels    -> disease labels
data/metadata  -> band names, wavelengths, CRS, dates
src/           -> reusable python modules
scripts/       -> run karne wali scripts
notebooks/     -> EDA aur experiments
daily_log/     -> day_01.md ... day_20.md (roz bharna hai)
docs/          -> data dictionary, report template, mentor questions
reports/       -> final report, figures
progress_tracker.csv -> 20 din ka tracker
```

## Phases
- **Phase 1 (Day 1-10):** Collection, inspection, cleaning
- **Phase 2 (Day 11-20):** Preprocessing, dataset preparation, documentation, handover

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
git init && git add . && git commit -m "Initial project structure"
```

## Pipeline run
```bash
python scripts/run_pipeline.py --config config/config.yaml
```

## Rules
1. `data/raw` ko kabhi modify nahi karna.
2. Har step script me ho, manual nahi.
3. Random seed hamesha 42.
4. Train/val/test split **spatial** hoga, random pixel nahi.
5. Roz ka log `daily_log/` me bharo aur git commit karo.
