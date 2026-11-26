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
# ============================================================
# Models
# ============================================================
class Zero:
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))


class MeanDelta:
    def fit(self, X, y):
        self.m = float(np.mean(y)); return self
    def predict(self, X): return np.full(len(X), self.m)


class DoyProfile:
    def fit(self, X, y):
        b = (X["doy"].values // 30)
        self.t = pd.Series(np.asarray(y)).groupby(b).mean().to_dict()
        self.g = float(np.mean(y)); return self
    def predict(self, X): return np.array([self.t.get(int(v // 30), self.g) for v in X["doy"].values])


ridge = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0))
rf = lambda: make_pipeline(SimpleImputer(strategy="median"),
                           RandomForestRegressor(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1))
hgb = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)

cfgs = [("persistence (delta=0)", Zero, ["ndvi"], False),
        ("train mean delta", MeanDelta, ["ndvi"], False),
        ("seasonal profile (doy)", DoyProfile, ["doy"], False),
        ("ridge NDVI-only", ridge, NDVI_F, True),
        ("random forest NDVI-only", rf, NDVI_F, True),
        ("hist-GB NDVI-only", hgb, NDVI_F, True)]
if WEATHER:
    cfgs += [("ridge NDVI+weather", ridge, NDVI_F + WEATHER, True),
             ("random forest NDVI+weather", rf, NDVI_F + WEATHER, True),
             ("hist-GB NDVI+weather", hgb, NDVI_F + WEATHER, True)]
    # ============================================================
# 1. Evaluate (val: fit on purged train | test: fit on purged train+val)
# ============================================================
print("\n===== Regression: next-observation NDVI change =====")
rows, store = [], {}
for name, mk, feats, learned in cfgs:
    mv = mk().fit(tr_p[feats], tr_p[TARGET])
    pv = mv.predict(va[feats])
    mt = mk().fit(trva_p[feats], trva_p[TARGET])
    pt = mt.predict(te[feats])
    store[name] = (pt, mt, feats)
    rows.append({"model": name, "learned": learned, "uses_weather": any(w in feats for w in WEATHER),
                 "val_mae": mean_absolute_error(va[TARGET], pv),
                 "test_mae": mean_absolute_error(te[TARGET], pt),
                 "test_rmse": float(np.sqrt(mean_squared_error(te[TARGET], pt))),
                 "test_r2": r2_score(te[TARGET], pt)})
res = pd.DataFrame(rows)
persist_mae = float(res.loc[res["model"] == "persistence (delta=0)", "test_mae"].iloc[0])
res["test_skill_vs_persistence"] = 1 - res["test_mae"] / persist_mae
res.round(4).to_csv("data/metadata/forecast_results.csv", index=False)
print(res.drop(columns=["learned"]).round(4).to_string(index=False))
print("(skill > 0 means better than 'no change'; r2 < 0 means worse than predicting the mean)")

learned_df = res[res["learned"]]
best = learned_df.sort_values("val_mae").iloc[0]["model"]        # chosen on VALIDATION, not test
print(f"\nBest learned model by validation MAE: {best}")


def boot_diff(df, ea, eb, n=1000):
    d = pd.DataFrame({"date": df["date"].values, "diff": np.asarray(ea) - np.asarray(eb)})
    g = d.groupby("date")["diff"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(SEED)
    out = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(n))]
    return float(np.mean(np.asarray(ea) - np.asarray(eb))), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


y_te = te[TARGET].values
err = {k: np.abs(v[0] - y_te) for k, v in store.items()}
md, lo, hi = boot_diff(te, err[best], err["persistence (delta=0)"])
beat = hi < 0
print(f"MAE difference (best - persistence): {md:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
      + ("-> significantly better than persistence" if beat else "-> NOT clearly better than persistence"))

weather_effect = None
if WEATHER:
    pair = [("random forest NDVI-only", "random forest NDVI+weather"), ("hist-GB NDVI-only", "hist-GB NDVI+weather"),
            ("ridge NDVI-only", "ridge NDVI+weather")]
    print("\nDoes weather help? (MAE with weather - MAE without; negative = helps)")
    weather_effect = {}
    for a, b in pair:
        d, l, h = boot_diff(te, err[b], err[a])
        weather_effect[b] = (d, l, h)
        print(f"  {b:32s} {d:+.4f}  95% CI [{l:+.4f}, {h:+.4f}]  " + ("helps" if h < 0 else ("hurts" if l > 0 else "no clear effect")))
        # ============================================================
# 2. Expanding-window CV over dates (train+val pool)
# ============================================================
print("\n===== Expanding-window CV (purged) =====")
pool = pd.concat([tr, va])
dts = sorted(pool["date"].unique())
chunks = np.array_split(dts, 5)
cv_cfg = [c for c in cfgs if c[0] in ("persistence (delta=0)", "random forest NDVI-only", "random forest NDVI+weather")]
cvr = []
for k in range(1, 5):
    test_d = set(chunks[k])
    tst = pool[pool["date"].isin(test_d)]
    trn = pool[pool["date"].isin(set(np.concatenate(chunks[:k])))]
    trn = trn[trn["next_date"] <= tst["date"].min()]
    if len(trn) < 50 or len(tst) < 10:
        continue
    row = {"fold": k, "train_rows": len(trn), "test_rows": len(tst)}
    for name, mk, feats, _ in cv_cfg:
        row[name] = mean_absolute_error(tst[TARGET], mk().fit(trn[feats], trn[TARGET]).predict(tst[feats]))
    cvr.append(row)
if cvr:
    cvd = pd.DataFrame(cvr)
    print(cvd.round(4).to_string(index=False))
    print("Mean MAE over folds:", cvd[[c[0] for c in cv_cfg]].mean().round(4).to_dict())
    cvd.to_csv("data/metadata/forecast_cv.csv", index=False)
else:
    print("Not enough data for CV folds.")