"""Day 23: validate the final dataset with baseline models + leakage demo + week 3 write-up."""
import os, sys, json, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS =====================
SEED = 42
N_BOOT = 300
TOP_BANDS = 15
# ====================================================
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "scikit-learn", "h5py", "pandas"])

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

H5 = "data/processed/terraspectra_dataset.h5"
if not os.path.exists(H5):
    raise SystemExit("Final dataset not found. Run day16_17_18_all.py first (needs labels).")
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("data/metadata", exist_ok=True)

meta = json.load(open("data/metadata/dataset_metadata.json")) if os.path.exists("data/metadata/dataset_metadata.json") else {}
scene = json.load(open("data/metadata/scene_metadata.json")) if os.path.exists("data/metadata/scene_metadata.json") else {}
dataset_name = scene.get("dataset", "unknown")
is_test = "Indian" in dataset_name
class_names = {int(k): v for k, v in meta.get("classes", {}).items()}
