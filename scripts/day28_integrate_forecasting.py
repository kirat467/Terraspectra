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
