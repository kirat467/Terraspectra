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
# ============================================================
# 3. PPT: 2 slides
# ============================================================
NAVY, GREY = RGBColor(0x1B, 0x2A, 0x5C), RGBColor(0x44, 0x44, 0x44)


def add_text(s, text, x, y, w, h, size=18, bold=False, color=GREY):
    tf = s.shapes.add_textbox(PI(x), PI(y), PI(w), PI(h)).text_frame
    tf.word_wrap = True
    for k, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.bold, p.font.color.rgb = PPt(size), bold, color
        p.space_after = PPt(6)


def add_fig(s, name, x, y, w):
    p = f"reports/figures/{name}.png"
    if os.path.exists(p):
        s.shapes.add_picture(p, PI(x), PI(y), width=PI(w))


if os.path.exists(PPTX):
    prs = Presentation(PPTX)
    for sl in list(prs.slides):                          # refresh: delete old forecasting slides
        titles = [sh.text_frame.text for sh in sl.shapes if sh.has_text_frame]
        if any(t.startswith(("5. Forecasting", "6. Forecasting")) for t in titles):
            for sldId in list(prs.slides._sldIdLst):
                if prs.part.related_part(sldId.rId) is sl.part:
                    prs.part.drop_rel(sldId.rId)
                    prs.slides._sldIdLst.remove(sldId)
    blank = prs.slide_layouts[6]
    s = prs.slides.add_slide(blank)
    add_text(s, "5. Forecasting data: NDVI time series + weather", 0.6, 0.35, 12, 0.9, 30, True, NAVY)
    add_text(s, f"Sentinel-2 NDVI, {summ.get('scenes_used')} clear dates\nCloud/shadow removed with SCL\n"
                f"Weather: NASA POWER rolling features (backward-looking)\nTable: {summ.get('table_rows')} rows, temporal split\n"
                "Target = next NDVI change (stress proxy)", 0.6, 1.5, 4.8, 4.5, 18)
    add_fig(s, "d26_ndvi_timeseries", 5.6, 1.6, 7.3)
    s = prs.slides.add_slide(blank)
    add_text(s, "6. Forecasting baselines and honest findings", 0.6, 0.35, 12, 0.9, 30, True, NAVY)
    add_text(s, "\n".join(["- " + v for v in verdict[:3]] + ["- Targets are proxies; disease labels still needed"]), 0.6, 1.5, 5.4, 5, 16)
    add_fig(s, "d27_model_comparison", 6.2, 1.6, 6.8)
    prs.save(PPTX)
    print("Updated", PPTX, "(+2 slides)")
else:
    print("PPT not found (run day19_20_all.py first); skipping")
# ============================================================
# 4. Forecasting loader + handover zip
# ============================================================
LOADER = r'''
"""Load the forecasting table. Run from the folder with forecasting_table.csv."""
import pandas as pd

NDVI_FEATURES = ["ndvi", "ndvi_prev", "ndvi_delta", "days_since_prev", "doy", "ndvi_std"]
WEATHER_FEATURES = ["temp_7d", "temp_range_7d", "humidity_7d", "rain_7d", "rain_14d",
                    "solar_7d", "favourable_7d", "favourable_14d"]
TARGET_REG, TARGET_CLS = "target_delta", "target_drop"
# NEVER use as features (they contain future information): ndvi_next, days_to_next, target_delta, target_drop


def load(path="forecasting_table.csv"):
    t = pd.read_csv(path, parse_dates=["date"])
    t["next_date"] = t["date"] + pd.to_timedelta(t["days_to_next"], unit="D")
    return t


def temporal_splits(t):
    """Train/val/test by date. Purge rows whose next observation falls in the evaluation period."""
    tr, va, te = (t[t["split"] == s] for s in ("train", "val", "test"))
    tr_purged = tr[tr["next_date"] <= va["date"].min()]
    trva_purged = pd.concat([tr, va])
    trva_purged = trva_purged[trva_purged["next_date"] <= te["date"].min()]
    return tr_purged, va, te, trva_purged


if __name__ == "__main__":
    t = load()
    tr, va, te, trva = temporal_splits(t)
    feats = [c for c in NDVI_FEATURES + WEATHER_FEATURES if c in t.columns]
    print("rows", len(t), "| train", len(tr), "val", len(va), "test", len(te))
    print("features:", feats)
    print("Baseline to beat: persistence (predict delta = 0). Report MAE with a date-block bootstrap CI.")
'''
open("handover/load_forecasting_example.py", "w", encoding="utf-8").write(LOADER.lstrip("\n"))

zp = "outputs/TerraSpectra_Dataset_Handover.zip"
old = {}
if os.path.exists(zp):
    with zipfile.ZipFile(zp) as z:
        old = {n: z.read(n) for n in z.namelist()}
extra = {
    "data/processed/forecasting_table.csv": "forecasting_table.csv",
    "data/processed/ndvi_timeseries_long.csv": "ndvi_timeseries_long.csv",
    "data/metadata/weather_daily_timeseries.csv": "weather_daily_timeseries.csv",
    "data/metadata/forecast_results.csv": "forecast_results.csv",
    "data/metadata/forecast_cv.csv": "forecast_cv.csv",
    "docs/day27_forecast_baselines.md": "forecast_baselines.md",
    "handover/load_forecasting_example.py": "load_forecasting_example.py",
    REPORT: "TerraSpectra_Project_Report.docx",
    PPTX: "TerraSpectra_DataPrep_Section.pptx",
}
for src, arc in extra.items():
    if os.path.exists(src):
        with open(src, "rb") as f:
            old[f"TerraSpectra_Handover/{arc}"] = f.read()
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for n, b in old.items():
        z.writestr(n, b)
print(f"Rebuilt {zp}: {len(old)} files, {os.path.getsize(zp) / 1e6:.1f} MB")