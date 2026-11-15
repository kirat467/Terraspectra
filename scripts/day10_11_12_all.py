import os, sys, glob, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

import numpy as np
import pandas as pd
import yaml
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)
os.makedirs("data/interim", exist_ok=True)
cfg = yaml.safe_load(open("config/config.yaml"))


def git(msg, tag=None):
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", msg], check=False)
        if tag:
            subprocess.run(["git", "tag", "-f", tag], check=True)
        if subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip():
            subprocess.run(["git", "push"], check=False)
            if tag:
                subprocess.run(["git", "push", "-f", "origin", tag], check=False)
    except Exception as e:
        print("Git issue:", e)


def mark(days, hours="3"):
    if os.path.exists("scripts/mark_done.py"):
        for d in days:
            subprocess.run([sys.executable, "scripts/mark_done.py", str(d), hours], stdout=subprocess.DEVNULL)


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else {}
