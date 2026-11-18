import os, sys, json, glob, ast, hashlib, shutil, subprocess, zipfile
from datetime import date
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (change here) =====================
RUN_REPRO_CHECK = True          # re-runs day16_17_18_all.py and compares file hashes (takes a minute)
AUTHOR = "Kailash Kumar"
REF_NO = "ITS/DSML/1513"
# ==================================================================