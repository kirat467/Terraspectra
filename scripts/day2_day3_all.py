import os, sys, csv, subprocess, urllib.request
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())
# ===================== SETTINGS (yahan badlo) =====================
BBOX = (84.0, 25.0, 86.0, 26.5)          # (west, south, east, north) - apna region daalo
DATE_RANGE = ("2024-01-01", "2025-12-31")
# ==================================================================
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def log_source(name, n_scenes, note):
    path = "docs/source_log.csv"
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "source", "n_scenes", "note"])
        w.writerow([date.today(), name, n_scenes, note])
# ===================== DAY 2: DATA SOURCES =====================
print("\n===== DAY 2: Hyperspectral data sources =====")
print("MANUAL KAAM (browser me register karo):")
print("  1. NASA Earthdata : https://urs.earthdata.nasa.gov   (EMIT)")
print("  2. DLR EOWEB      : https://eoweb.dlr.de             (EnMAP, DESIS)")
print("  3. ASI PRISMA     : https://prisma.asi.it            (PRISMA)")
print("  4. USGS Earth Explorer : https://earthexplorer.usgs.gov (Hyperion archive)")

try:
    import earthaccess
    print("\nEMIT scenes search ho rahi hai (Earthdata login poochega)...")
    earthaccess.login()
    results = earthaccess.search_data(
        short_name="EMITL2ARFL",
        bounding_box=BBOX,
        temporal=DATE_RANGE,
        count=20,
    )
    print("EMIT scenes mile:", len(results))
    names = [r["umm"]["GranuleUR"] for r in results]
    for n in names[:10]:
        print("  ", n)
    with open("docs/emit_scenes_found.txt", "w") as f:
        f.write("\n".join(names))
    log_source("EMIT L2A", len(results), f"bbox={BBOX}, dates={DATE_RANGE}")
except Exception as e:
    print("EMIT search skip hui:", e)
    print("(Earthdata account/login ya internet check karo. Baaki kaam chalega.)")
    log_source("EMIT L2A", "NA", f"search failed: {e}")
    # ===================== DAY 3: LABEL DATASETS + INDIAN PINES =====================
print("\n===== DAY 3: Label datasets + Indian Pines test =====")
# label datasets tracking sheet
lab_path = "docs/label_datasets.csv"
if not os.path.exists(lab_path):
    with open(lab_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "crop", "disease", "link", "licence", "size", "usable"])
        w.writerow(["(Kaggle me search karo: hyperspectral crop disease)", "", "", "", "", "", ""])
        w.writerow(["(Zenodo me search karo: hyperspectral plant disease)", "", "", "", "", "", ""])
        w.writerow(["(Mendeley Data me search karo: hyperspectral leaf disease)", "", "", "", "", "", ""])
    print("Created docs/label_datasets.csv (isme kam se kam 3 datasets khud bharo)")