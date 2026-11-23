"""Day 25: unit tests + run_all.py + lint + data manifest + README + tag."""
import os, sys, json, glob, hashlib, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "ruff", "h5py", "numpy", "scipy"])

os.makedirs("tests", exist_ok=True)
os.makedirs("docs", exist_ok=True)
os.makedirs("logs", exist_ok=True)