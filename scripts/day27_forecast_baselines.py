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
    # ============================================================
# 3. Stress-proxy classification (next NDVI drop)
# ============================================================
print("\n===== Classification: next-NDVI drop (stress PROXY, not disease) =====")
cls_summary = None
if "target_drop" in tab.columns and WEATHER is not None:
    cf = NDVI_F + WEATHER
    trc, vac, tec = tr_p.dropna(subset=["target_drop"]), va.dropna(subset=["target_drop"]), te.dropna(subset=["target_drop"])
    pos_tr, pos_te = int(trc["target_drop"].sum()), int(tec["target_drop"].sum())
    print(f"Positives: train {pos_tr}/{len(trc)} | test {pos_te}/{len(tec)}")
    if pos_tr >= 5 and pos_te >= 3 and trc["target_drop"].nunique() == 2:
        mk_c = lambda: make_pipeline(SimpleImputer(strategy="median"),
                                     RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample",
                                                            random_state=SEED, n_jobs=-1))
        cv_model = mk_c().fit(trc[cf], trc["target_drop"])
        pv = cv_model.predict_proba(vac[cf])[:, 1]
        thr = 0.5
        if vac["target_drop"].nunique() == 2:
            p_, r_, t_ = precision_recall_curve(vac["target_drop"], pv)
            f1s = 2 * p_[:-1] * r_[:-1] / (p_[:-1] + r_[:-1] + 1e-9)
            thr = float(t_[int(np.argmax(f1s))])
        trva_c = pd.concat([tr, va]).query("next_date <= @test_start").dropna(subset=["target_drop"])
        fm = mk_c().fit(trva_c[cf], trva_c["target_drop"])
        pt_ = fm.predict_proba(tec[cf])[:, 1]
        yb = tec["target_drop"].values
        pred = pt_ >= thr
        tp = int(((pred == 1) & (yb == 1)).sum())
        prec, rec = tp / max(pred.sum(), 1), tp / max(yb.sum(), 1)
        mom = average_precision_score(yb, -tec["ndvi_delta"].values)
        cls_summary = {"prevalence_test": float(yb.mean()), "pr_auc_model": float(average_precision_score(yb, pt_)),
                       "pr_auc_momentum_baseline": float(mom), "roc_auc_model": float(roc_auc_score(yb, pt_)),
                       "threshold_from_val": thr, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec + 1e-9)}
        print(json.dumps({k: round(v, 3) for k, v in cls_summary.items()}, indent=2))
        print("PR-AUC should be clearly above the prevalence (random) and the momentum baseline to be useful.")
    else:
        print("Too few positives for a reliable classifier (need >=5 train and >=3 test). Skipping.")
# ============================================================
# 4. Importance + figures
# ============================================================
imp_df = None
if best in store:
    _, bm, bf = store[best]
    pi = permutation_importance(bm, te[bf], te[TARGET], n_repeats=20, random_state=SEED, scoring="neg_mean_absolute_error")
    imp_df = pd.DataFrame({"feature": bf, "mae_increase": pi.importances_mean, "std": pi.importances_std}).sort_values("mae_increase", ascending=False)
    imp_df.to_csv("data/metadata/forecast_feature_importance.csv", index=False)
    print("\nPermutation importance (test, descriptive only):\n", imp_df.head(10).round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.5))
o = res.sort_values("test_mae")
ax.barh(o["model"], o["test_mae"], color=["tab:green" if m == best else "tab:gray" for m in o["model"]])
ax.axvline(persist_mae, color="red", ls="--", label="persistence")
ax.set_xlabel("Test MAE (NDVI change)"); ax.set_title("Forecasting baselines (green = best on validation)"); ax.legend()
plt.tight_layout(); plt.savefig("reports/figures/d27_model_comparison.png", dpi=150); plt.close()

pt_best = store[best][0]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].scatter(y_te, pt_best, s=12, alpha=0.5)
lim = [min(y_te.min(), pt_best.min()), max(y_te.max(), pt_best.max())]
ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("actual NDVI change"); ax[0].set_ylabel("predicted"); ax[0].set_title(f"{best}: test")
dd = te.assign(pred=pt_best).groupby("date")[[TARGET, "pred"]].mean()
ax[1].plot(dd.index, dd[TARGET], "o-", label="actual (mean of cells)"); ax[1].plot(dd.index, dd["pred"], "s--", label="predicted")
ax[1].axhline(0, color="gray", lw=0.8); ax[1].set_title("Test period: mean NDVI change per date"); ax[1].legend()
plt.setp(ax[1].get_xticklabels(), rotation=30)
plt.tight_layout(); plt.savefig("reports/figures/d27_predictions.png", dpi=150); plt.close()

if imp_df is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    t10 = imp_df.head(10).iloc[::-1]
    ax.barh(t10["feature"], t10["mae_increase"], xerr=t10["std"])
    ax.set_xlabel("MAE increase when shuffled"); ax.set_title("Feature importance (test, descriptive)")
    plt.tight_layout(); plt.savefig("reports/figures/d27_feature_importance.png", dpi=150); plt.close()
# ============================================================
# 5. Verdict + docs
# ============================================================
best_row = res[res["model"] == best].iloc[0]
verdict = [f"Best learned model (chosen on validation): {best}; test MAE {best_row['test_mae']:.4f} vs persistence {persist_mae:.4f} "
           f"(skill {best_row['test_skill_vs_persistence']:+.2%}, R2 {best_row['test_r2']:.3f})."]
verdict.append("It is clearly better than persistence (bootstrap CI excludes 0)." if beat else
               "It is NOT clearly better than persistence; with this amount of data NDVI(t) alone is a hard baseline to beat.")
if weather_effect:
    helped = [k for k, (d, l, h) in weather_effect.items() if h < 0]
    verdict.append("Weather features gave a measurable improvement for: " + ", ".join(helped) + "." if helped
                   else "Weather features gave no statistically clear improvement.")
for v in verdict:
    print("-", v)

with open("docs/day27_forecast_baselines.md", "w", encoding="utf-8") as f:
    f.write(f"# Day 27: forecasting baselines ({date.today()})\n\n")
    f.write("Target: next-observation NDVI change (regression) and NDVI drop (classification). **Stress proxies, not disease labels.**\n\n")
    f.write("Protocol: temporal split, training rows purged if their target looked into the evaluation period, best model chosen on validation, "
            "date-block bootstrap CIs (cells are spatially correlated, so independent information = number of dates).\n\n")
    f.write("| model | weather | val MAE | test MAE | test RMSE | test R2 | skill vs persistence |\n|---|---|---|---|---|---|---|\n")
    for r in res.itertuples():
        f.write(f"| {r.model} | {r.uses_weather} | {r.val_mae:.4f} | {r.test_mae:.4f} | {r.test_rmse:.4f} | {r.test_r2:.3f} | {r.test_skill_vs_persistence:+.2%} |\n")
    f.write("\n## Verdict\n" + "\n".join(f"- {v}" for v in verdict) + "\n")
    if cls_summary:
        f.write("\n## Classification (stress proxy)\n" + "\n".join(f"- {k}: {v:.3f}" for k, v in cls_summary.items()) + "\n")
    f.write("\n## Limitations\n- Few dates and one small area: wide uncertainty.\n- NDVI change can also come from harvest, irrigation, drought; not only disease.\n"
            "- Real disease labels are needed for a true disease-forecasting claim.\n")

with open("docs/forecast_writeup.txt", "w", encoding="utf-8") as f:
    f.write("Built forecasting baselines on the Sentinel-2 NDVI + NASA POWER weather table using a temporal split with leakage purging. "
            + verdict[0] + " " + verdict[1] + (" " + verdict[2] if weather_effect else "")
            + " Targets are NDVI-change stress proxies; confirmed disease labels are still required for disease forecasting.")
print("\nSaved docs/day27_forecast_baselines.md, docs/forecast_writeup.txt, data/metadata/forecast_results.csv")

try:
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "Day 27: forecasting baselines with temporal split and bootstrap CIs"], check=False)
    if subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip():
        subprocess.run(["git", "push"], check=False)
except Exception as e:
    print("Git issue:", e)
print("\nDAY 27 DONE. Figures: d27_model_comparison, d27_predictions, d27_feature_importance")