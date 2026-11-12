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
    # Indian Pines download
os.makedirs("data/raw/indian_pines", exist_ok=True)
files = {
    "Indian_pines_corrected.mat": "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
    "Indian_pines_gt.mat": "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
}
ok = True
for name, url in files.items():
    path = f"data/raw/indian_pines/{name}"
    if not os.path.exists(path):
        try:
            print("Downloading", name, "...")
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            ok = False
            print("Download fail:", name, e)
            print("  Manually download karke data/raw/indian_pines/ me rakho.")

if ok:
    cube = sio.loadmat("data/raw/indian_pines/Indian_pines_corrected.mat")["indian_pines_corrected"]
    gt = sio.loadmat("data/raw/indian_pines/Indian_pines_gt.mat")["indian_pines_gt"]

    print("Cube (H,W,B):", cube.shape, "| GT:", gt.shape)
    classes, counts = np.unique(gt, return_counts=True)
    print("Pixels per class:", dict(zip(classes.tolist(), counts.tolist())))

    cube = np.transpose(cube, (2, 0, 1)).astype("float32")   # (B,H,W)
    os.makedirs("data/interim", exist_ok=True)
    np.save("data/interim/indian_pines_cube.npy", cube)
    np.save("data/interim/indian_pines_gt.npy", gt.astype("int32"))

    os.makedirs("reports/figures", exist_ok=True)
    plt.figure(figsize=(9, 5))
    for c in classes:
        if c == 0:
            continue
        plt.plot(cube[:, gt == c].mean(axis=1), label=f"class {c}")
    plt.xlabel("Band index"); plt.ylabel("Mean value")
    plt.title("Indian Pines: class-wise mean spectra")
    plt.legend(ncol=4, fontsize=7); plt.tight_layout()
    plt.savefig("reports/figures/indian_pines_spectra.png", dpi=150)
    plt.close()

    # NaN/range check + band means figure
    print("NaN count:", int(np.isnan(cube).sum()), "| min/max:", float(cube.min()), float(cube.max()))
    plt.figure(figsize=(6, 5))
    plt.imshow(gt, cmap="nipy_spectral"); plt.colorbar(label="class")
    plt.title("Indian Pines ground truth"); plt.tight_layout()
    plt.savefig("reports/figures/indian_pines_gt.png", dpi=150)
    plt.close()
    print("Saved figures in reports/figures/")
    log_source("Indian Pines", 1, "benchmark for pipeline testing, 200 bands")
