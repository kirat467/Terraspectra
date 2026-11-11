"""Quick inspection: python scripts/inspect_scene.py data/raw/scene.tif"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.io_utils import load_cube, describe

cube, profile, wl = load_cube(sys.argv[1])
describe(cube, wl)
x = wl if wl is not None else np.arange(cube.shape[0])
plt.figure(figsize=(8, 4))
plt.plot(x, np.nanmean(cube, axis=(1, 2)))
plt.xlabel("Wavelength (nm)" if wl is not None else "Band"); plt.ylabel("Mean value")
plt.title("Mean spectrum"); plt.tight_layout()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.savefig("reports/figures/mean_spectrum.png", dpi=150)
print("Saved reports/figures/mean_spectrum.png")
