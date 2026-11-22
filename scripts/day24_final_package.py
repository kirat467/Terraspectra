"""Day 24: demo app + Word project report + submission checklist + week 4 write-up."""
import os, sys, json, glob, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

AUTHOR = "Kailash Kumar"
REF_NO = "ITS/DSML/1513"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "streamlit", "python-docx", "pandas", "h5py", "matplotlib"])

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

os.makedirs("reports", exist_ok=True)
os.makedirs("docs", exist_ok=True)


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else {}


def cs(p):
    return pd.read_csv(p) if os.path.exists(p) else None
