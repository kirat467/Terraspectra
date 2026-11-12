import sys, subprocess, importlib, os
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
print("Working folder:", os.getcwd())

# ---------- 1. Packages install ----------
print("\n[1/5] Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
subprocess.run([sys.executable, "-m", "pip", "install", "earthaccess"])

# ---------- 2. Setup verify ----------
print("\n[2/5] Verifying setup...")
for m in ["numpy", "scipy", "pandas", "rasterio", "matplotlib", "h5py", "yaml", "sklearn", "earthaccess"]:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  FAIL {m}: {e}")
try:
    from osgeo import gdal
    print("  OK   GDAL", gdal.__version__)
except Exception:
    print("  INFO GDAL python binding nahi mila (zaroori nahi, rasterio kaafi hai abhi)")

# ---------- 3. Mentor message ----------
print("\n[3/5] Writing mentor message...")
msg = """Hello Sir/Ma'am,

I am Kailash Kumar (Ref: ITS/DSML/1513), working on TerraSpectra: Hyperspectral Crop Disease Forecasting. My role is Data Collection & Preparation. Kindly guide me on:

1. Will disease labels / field data be provided by the company? If not, can I use public datasets?
2. Target crop, disease and region?
3. Preferred sensor (EnMAP / PRISMA / EMIT / other)?
4. Required output format for the modeling team (.npy / HDF5 / .tif)?
5. Patch-level or pixel-level classification?
6. Is multi-date time series data needed for forecasting?
7. Storage / compute limits?
8. How often should I share progress updates?

Thank you.
Kailash Kumar
"""
with open("docs/mentor_message.txt", "w", encoding="utf-8") as f:
    f.write(msg)
print("  Saved docs/mentor_message.txt (isko copy karke mentor ko bhejo)")

# ---------- 4. Progress tracker helper + Day 1 mark ----------
print("\n[4/5] Progress tracker...")
helper = '''import sys
import pandas as pd
from datetime import date

day = int(sys.argv[1])
hours = sys.argv[2] if len(sys.argv) > 2 else ""
df = pd.read_csv("progress_tracker.csv", dtype=str).fillna("")
df.loc[df["Day"] == str(day), ["Status", "Date", "Hours"]] = ["Done", str(date.today()), hours]
df.to_csv("progress_tracker.csv", index=False)
print(df[["Day", "Topic", "Status"]].to_string(index=False))
'''
with open("scripts/mark_done.py", "w", encoding="utf-8") as f:
    f.write(helper)
subprocess.run([sys.executable, "scripts/mark_done.py", "1", "3"])

# ---------- 5. Git ----------
print("\n[5/5] Git...")
try:
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], check=True)
    subprocess.run(["git", "add", "."], check=True)
    subprocess.run(["git", "commit", "-m", "Day 1: project setup"], check=True)
    print("  Commit ho gaya.")
    print("  GitHub pe repo banao, fir terminal me ye 3 commands chalao:")
    print("    git branch -M main")
    print("    git remote add origin https://github.com/<username>/terraspectra-data.git")
    print("    git push -u origin main")
except Exception as e:
    print("  Git error:", e, "\n  Git installed hai? (git --version check karo)")

print("\nDay 1 DONE. Ab docs/mentor_message.txt mentor ko bhejo.")