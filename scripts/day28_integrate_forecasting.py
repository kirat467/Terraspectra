"""Day 28: integrate forecasting results (Day 26-27) into report, PPT, handover, tests, checklist."""
import os, sys, json, zipfile, glob, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-docx", "python-pptx", "pytest", "pandas"])

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from pptx import Presentation
from pptx.util import Inches as PI, Pt as PPt
from pptx.dml.color import RGBColor

AUTHOR, REF_NO = "Kailash Kumar", "ITS/DSML/1513"
REPORT = "reports/TerraSpectra_Project_Report.docx"
PPTX = "reports/TerraSpectra_DataPrep_Section.pptx"
for d in ("docs", "tests", "handover", "outputs"):
    os.makedirs(d, exist_ok=True)


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def cs(p):
    return pd.read_csv(p) if os.path.exists(p) else None


def md_section(path, title):
    if not os.path.exists(path):
        return []
    txt = open(path, encoding="utf-8").read()
    if title not in txt:
        return []
    part = txt.split(title, 1)[1].split("\n## ", 1)[0]
    return [l[2:].strip() for l in part.splitlines() if l.startswith("- ")]
# ---------- inputs from Day 26-27 ----------
res = cs("data/metadata/forecast_results.csv")
cv = cs("data/metadata/forecast_cv.csv")
imp = cs("data/metadata/forecast_feature_importance.csv")
summ = jl("data/metadata/timeseries_summary.json")
if res is None or not os.path.exists("data/processed/forecasting_table.csv"):
    raise SystemExit("Day 26/27 outputs not found. Run day26_ndvi_timeseries.py and day27_forecast_baselines.py first.")
verdict = md_section("docs/day27_forecast_baselines.md", "## Verdict")
cls_lines = md_section("docs/day27_forecast_baselines.md", "## Classification (stress proxy)")
learned = res[res["learned"]].sort_values("val_mae")
best = learned.iloc[0]
persist = res[res["model"] == "persistence (delta=0)"].iloc[0]
has_weather = bool(res["uses_weather"].any())
print(f"Best learned model (validation): {best['model']} | test MAE {best['test_mae']:.4f} vs persistence {persist['test_mae']:.4f}")
# ============================================================
# 1. TESTS for the forecasting table
# ============================================================
TEST = r'''
"""Checks for the forecasting table (temporal ordering, target correctness). Run: pytest tests -q"""
import os
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "data", "processed", "forecasting_table.csv")
L = os.path.join(ROOT, "data", "processed", "ndvi_timeseries_long.csv")
pytestmark = pytest.mark.skipif(not os.path.exists(T), reason="forecasting table not built yet")


def tab():
    return pd.read_csv(T, parse_dates=["date"])


def test_temporal_split_is_ordered():
    t = tab()
    tr, va, te = (t[t["split"] == s]["date"] for s in ("train", "val", "test"))
    assert len(tr) and len(va) and len(te)
    assert tr.max() < va.min() <= va.max() < te.min()


def test_ndvi_range_and_no_missing_targets():
    t = tab()
    assert t["ndvi"].between(-1, 1).all()
    assert t["target_delta"].notna().all() and t["ndvi_prev"].notna().all()
    assert (t["days_to_next"] > 0).all()


@pytest.mark.skipif(not os.path.exists(L), reason="long NDVI file missing")
def test_target_equals_next_observation_minus_now():
    t = tab()
    lg = pd.read_csv(L, parse_dates=["date"]).sort_values(["cell", "date"])
    lg["next"] = lg.groupby("cell")["ndvi"].shift(-1)
    m = t.merge(lg[["cell", "date", "next"]], on=["cell", "date"])
    assert len(m) == len(t)
    assert np.allclose(m["ndvi_next"], m["next"], atol=1e-6)
    assert np.allclose(m["target_delta"], m["ndvi_next"] - m["ndvi"], atol=1e-6)
'''
open("tests/test_forecasting.py", "w", encoding="utf-8").write(TEST.lstrip("\n"))
print("\n===== Running all tests =====")
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--tb=short", "-rs"], capture_output=True, text=True)
open("docs/test_report.txt", "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
print("\n".join(r.stdout.strip().splitlines()[-15:]))
tests_ok = r.returncode == 0
print("TESTS:", "ALL PASSED" if tests_ok else "SOME FAILED (see docs/test_report.txt)")
# ============================================================
# 2. WORD REPORT: Appendix A
# ============================================================
if os.path.exists(REPORT):
    doc = Document(REPORT)
    marker = "Appendix A"
    start = next((p for p in doc.paragraphs if p.text.startswith(marker)), None)
    if start is not None:                        # refresh: remove old appendix
        el = start._element
        body, sib = el.getparent(), el.getnext()
        body.remove(el)
        while sib is not None and not sib.tag.endswith("sectPr"):
            nxt = sib.getnext()
            body.remove(sib)
            sib = nxt

    def table(df, max_rows=20):
        if df is None or len(df) == 0:
            return
        df = df.head(max_rows)
        t = doc.add_table(rows=1, cols=len(df.columns))
        try:
            t.style = "Light Grid Accent 1"
        except Exception:
            t.style = "Table Grid"
        for i, c in enumerate(df.columns):
            t.rows[0].cells[i].text = str(c)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for i, v in enumerate(row.values):
                cells[i].text = f"{v:.4g}" if isinstance(v, float) else str(v)
        doc.add_paragraph()

    def fig(name, cap, w=6.0):
        p = f"reports/figures/{name}.png"
        if os.path.exists(p):
            doc.add_picture(p, width=Inches(w))
            doc.add_paragraph(cap).runs[0].italic = True

    def bullets(items):
        for x in items:
            doc.add_paragraph(str(x), style="List Bullet")

    doc.add_heading("Appendix A: Forecasting data and baselines (Day 26-27)", 1)
    doc.add_paragraph("To support the forecasting part of the project, a time series of vegetation greenness (NDVI) and weather was built, "
                      "and simple baselines were evaluated on it.")
    doc.add_heading("A.1 Data", 2)
    bullets([f"Sentinel-2 L2A (Microsoft Planetary Computer); {summ.get('scenes_used')} dates between {summ.get('date_range')}",
             "Cloud, shadow, snow and water pixels removed with the scene classification layer (SCL); NDVI from B04/B08",
             f"{summ.get('observations')} cell-date observations on a grid of cells over the study area",
             "Weather: NASA POWER daily data converted to backward-looking rolling features (7 and 14 days); no future information used" if has_weather else "Weather features were not available",
             f"Forecasting table: {summ.get('table_rows')} rows; temporal split {summ.get('split_counts')}",
             "Target: NDVI change at the next observation (regression) and NDVI drop > "
             f"{summ.get('drop_threshold')} (classification). These are stress proxies, NOT disease labels."])
    doc.add_heading("A.2 Evaluation protocol", 2)
    bullets(["Temporal split (train = earliest dates, test = latest dates)",
             "Training rows whose target looked into the evaluation period were purged (no look-ahead)",
             "Best model chosen on validation error, not test error",
             "Confidence intervals by bootstrapping dates (cells are spatially correlated, so dates are the independent units)"])
    doc.add_heading("A.3 Results", 2)
    cols = ["model", "uses_weather", "val_mae", "test_mae", "test_r2", "test_skill_vs_persistence"]
    table(res[[c for c in cols if c in res.columns]].round(4))
    bullets(verdict)
    fig("d27_model_comparison", "Figure A1: Test MAE of baselines")
    fig("d27_predictions", "Figure A2: Predictions on the test period")
    fig("d26_ndvi_timeseries", "Figure A3: NDVI time series per grid cell with rainfall")
    fig("d26_ndvi_heatmap", "Figure A4: NDVI by cell and date")
    if cv is not None:
        doc.add_heading("A.4 Expanding-window cross-validation (MAE)", 2)
        table(cv.round(4))
    if imp is not None:
        doc.add_heading("A.5 Feature importance (descriptive)", 2)
        table(imp.round(4), 10)
        fig("d27_feature_importance", "Figure A5: Permutation importance")
    if cls_lines:
        doc.add_heading("A.6 Stress-proxy classification", 2)
        bullets(cls_lines)
    doc.add_heading("A.7 Limitations", 2)
    bullets(["Few dates and one small area: uncertainty is wide; persistence ('no change') is a strong baseline for smooth NDVI series.",
             "NDVI change can also come from harvest, irrigation or drought, not only disease.",
             "A true disease-forecasting claim needs confirmed disease labels (field data), which are still pending."])
    doc.save(REPORT)
    print("Updated", REPORT, "(Appendix A)")
else:
    print("Report not found (run day24_final_package.py first); skipping Word update")
