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