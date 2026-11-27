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