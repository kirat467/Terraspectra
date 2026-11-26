"""Day 27: forecasting baselines on the NDVI + weather table (temporal split, purged, bootstrap CI)."""
import os, sys, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "pandas", "matplotlib"])
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             average_precision_score, roc_auc_score, precision_recall_curve)
from sklearn.inspection import permutation_importance

SEED = 42
TARGET = "target_delta"
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)
os.makedirs("docs", exist_ok=True)
# ============================================================
# Load table from Day 26
# ============================================================
P = "data/processed/forecasting_table.csv"
if not os.path.exists(P):
    raise SystemExit("forecasting_table.csv not found. Run day26_ndvi_timeseries.py first.")
tab = pd.read_csv(P, parse_dates=["date"])
need = {"date", "cell", "ndvi", "ndvi_prev", "ndvi_delta", "days_since_prev", "days_to_next", "doy", TARGET, "split"}
if not need.issubset(tab.columns):
    raise SystemExit(f"Missing columns: {sorted(need - set(tab.columns))}")
tab["next_date"] = tab["date"] + pd.to_timedelta(tab["days_to_next"], unit="D")

NDVI_F = [c for c in ["ndvi", "ndvi_prev", "ndvi_delta", "days_since_prev", "doy", "ndvi_std"] if c in tab.columns]
WEATHER = [c for c in ["temp_7d", "temp_range_7d", "humidity_7d", "rain_7d", "rain_14d", "solar_7d",
                       "favourable_7d", "favourable_14d"] if c in tab.columns and tab[c].notna().mean() > 0.8]
# days_to_next is NOT a feature: it depends on future cloud cover (would leak)
print(f"Rows {len(tab)} | NDVI features {NDVI_F} | weather features {WEATHER or 'none'}")

tr, va, te = (tab[tab["split"] == s] for s in ("train", "val", "test"))
if min(len(tr), len(va), len(te)) < 10:
    raise SystemExit(f"Too few rows (train {len(tr)}, val {len(va)}, test {len(te)}). Add more scenes in day26 (longer DATE range / larger MAX_CLOUD).")
val_start, test_start = va["date"].min(), te["date"].min()
tr_p = tr[tr["next_date"] <= val_start]                                   # purge: no peeking into val period
trva_p = pd.concat([tr, va]).query("next_date <= @test_start")           # purge for the test fit
print(f"Train {len(tr)} (purged {len(tr_p)}) | val {len(va)} | test {len(te)} (final fit rows {len(trva_p)})")
print(f"Effective independent information = number of dates: train {tr['date'].nunique()}, val {va['date'].nunique()}, test {te['date'].nunique()}")