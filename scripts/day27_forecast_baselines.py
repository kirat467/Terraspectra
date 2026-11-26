"""Day 27: forecasting baselines on the NDVI + weather table (temporal split, purged, bootstrap CI)."""
import os, sys, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "pandas", "matplotlib"])