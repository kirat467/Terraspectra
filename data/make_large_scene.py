"""Generate a large simulated scene (default 1024 x 1024 pixels x 200 bands, about 15 km across at 15 m) without holding it in memory.

Why: the 135 x 135 farm is tiny next to a real flight line. This writes the cube in strips of rows straight into a memory-mapped
file, so working memory is one strip, not the whole cube plus temporaries. Measured at 1024 x 1024 (peak resident memory, which also counts
the memory-mapped file pages being written): strip 8 rows = 538 MB, 32 rows = 859 MB, 128 rows = 1959 MB. It is
also the test bed for the memory audit planned for Saturday.

Layout: a grid of fields with narrow tracks between them, each field a crop (some fallow), a few ponds, and several outbreaks that
grow with `--step` (0 to 5, like the six weekly scans). Labels use the same codes as make_scene.py: 0 healthy, 1 early stress,
2 visible stress, -1 not a crop.

Output in --out (default data/generated/, git-ignored):
    large_cube_<step>.npy    uint16 reflectance x 10000, shape (rows, cols, bands), memory-mappable with np.load(mmap_mode="r")
    large_labels_<step>.npy  int8 (rows, cols)
    large_summary_<step>.json  size, bands, class counts, seconds, peak memory

Results do not depend on --strip: the noise for each row comes from its own seed.

    python data/make_large_scene.py --size 1024 --step 5
    python data/make_large_scene.py --size 128 --step 3 --out /tmp/small     # quick check
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spectral_simulator import EARLY_STRESS, N_BANDS, VISIBLE_STRESS, leaf_spectrum, soil_spectrum, water_spectrum  # noqa: E402

SCALE = 10000
CROPS = ["Paddy", "Sugarcane", "Ragi", "Maize", "Fallow"]
CROP_NIR = {"Paddy": 1.00, "Sugarcane": 1.08, "Ragi": 0.94, "Maize": 1.04, "Fallow": 0.0}


def layout(size, seed=11):
    """Per-pixel crop index (-1 track, 0..4 crop) from jittered grid lines, plus a per-field NIR factor."""
    rng = np.random.default_rng(seed)
    n = max(2, size // 170)                                            # about 6 fields across at 1024
    edges = lambda: np.unique(np.clip(np.linspace(0, size, n + 1) + np.r_[0, rng.integers(-size // 25, size // 25 + 1, n - 1), 0], 0, size).astype(int))
    ys, xs = edges(), edges()
    track = max(2, size // 256)
    crop = np.full((size, size), -1, dtype=np.int8)
    nir = np.ones((size, size), dtype=np.float32)
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            k = int(rng.integers(0, len(CROPS)))
            y0, y1, x0, x1 = ys[i] + track, ys[i + 1] - track, xs[j] + track, xs[j + 1] - track
            if y1 > y0 and x1 > x0:
                crop[y0:y1, x0:x1] = k
                nir[y0:y1, x0:x1] = CROP_NIR[CROPS[k]] * (1 + 0.03 * rng.normal())
    return crop, nir


def outbreak_centres(size, seed=5, count=6):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0.1, 0.9) * size, rng.uniform(0.1, 0.9) * size, rng.uniform(0.6, 1.0), rng.uniform(0.02, 0.05) * size) for _ in range(count)]


def severity_rows(r0, r1, size, step, centres):
    """Disease severity 0..1 for rows r0:r1 at scan `step` (0..5): blobs that start small and faint, then grow."""
    s = step / 5.0
    yy, xx = np.mgrid[r0:r1, 0:size].astype(np.float32)
    out = np.zeros((r1 - r0, size), dtype=np.float32)
    for cx, cy, peak, rad in centres:
        radius = rad * (0.5 + 3.0 * s ** 1.1)
        out += (0.12 + 0.88 * s ** 1.3) * peak * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / radius ** 2))
    return np.clip(out, 0.0, 1.0)


def ponds(size, seed=9, count=8):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0.05, 0.95) * size, rng.uniform(0.05, 0.95) * size, rng.uniform(0.008, 0.02) * size, rng.uniform(0.006, 0.015) * size) for _ in range(count)]


def make_strip(r0, r1, size, step, crop, nir, centres, pond_list, seed):
    h = r1 - r0
    sev = severity_rows(r0, r1, size, step, centres)
    c, nr = crop[r0:r1], nir[r0:r1]
    in_field = c >= 0
    fallow = c == CROPS.index("Fallow")
    # noise: one generator per row, so the result is the same whatever the strip size
    noise = np.empty((h, size, N_BANDS), dtype=np.float32)
    bright = np.empty((h, size), dtype=np.float32)
    for i in range(h):
        rng = np.random.default_rng([seed, step, r0 + i])
        noise[i] = 0.004 * rng.normal(size=(size, N_BANDS))
        bright[i] = 1 + 0.04 * rng.normal(size=size)
    canopy = leaf_spectrum((sev * in_field).ravel(), nr.ravel(), bright.ravel()).astype(np.float32)
    soil = (soil_spectrum()[None, :] * bright.ravel()[:, None]).astype(np.float32)
    cube = soil.copy()
    crop_px = (in_field & ~fallow).ravel()
    cube[crop_px] = canopy[crop_px]
    fb = (in_field & fallow).ravel()
    cube[fb] = 0.8 * soil_spectrum() + 0.2 * leaf_spectrum(0.0)
    cube = cube.reshape(h, size, N_BANDS)
    yy, xx = np.mgrid[r0:r1, 0:size]
    water = np.zeros((h, size), dtype=bool)
    for cx, cy, rx, ry in pond_list:
        inside = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) < 1
        cube[inside] = water_spectrum()
        water |= inside
    cube += noise
    label = np.zeros((h, size), dtype=np.int8)
    label[sev >= EARLY_STRESS] = 1
    label[sev >= VISIBLE_STRESS] = 2
    label[~in_field | fallow | water] = -1                       # open water is not a crop (a pond inside an outbreak is not "stress")
    return np.round(np.clip(cube, 0, 1) * SCALE).astype(np.uint16), label


def peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux reports KB


def generate(out, size=1024, step=5, strip=16, seed=7, farm=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    crop, nir = layout(size, 11 + farm)                      # farm = 0, 1, 2 ... gives a different field layout, outbreaks and ponds
    centres, pond_list = outbreak_centres(size, 5 + farm), ponds(size, 9 + farm)
    seed = seed + 1000 * farm
    cube = np.lib.format.open_memmap(out / f"large_cube_{step}.npy", mode="w+", dtype=np.uint16, shape=(size, size, N_BANDS))
    labels = np.lib.format.open_memmap(out / f"large_labels_{step}.npy", mode="w+", dtype=np.int8, shape=(size, size))
    for r0 in range(0, size, strip):
        r1 = min(size, r0 + strip)
        cube[r0:r1], labels[r0:r1] = make_strip(r0, r1, size, step, crop, nir, centres, pond_list, seed)
    cube.flush(); labels.flush()
    counts = {n: int((labels == v).sum()) for n, v in [("healthy", 0), ("early_stress", 1), ("visible_stress", 2), ("not_crop", -1)]}
    summary = {"size": size, "bands": N_BANDS, "step": step, "strip_rows": strip, "cube_mb": round(cube.nbytes / 1e6, 1), "class_pixels": counts,
               "seconds": round(time.time() - t0, 1), "peak_memory_mb": round(peak_memory_mb(), 0), "simulated": True}
    (out / f"large_summary_{step}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(root / "data" / "generated"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--step", type=int, default=5, choices=range(6))
    ap.add_argument("--strip", type=int, default=16, help="rows per strip (smaller = less memory)")
    a = ap.parse_args()
    print(json.dumps(generate(a.out, a.size, a.step, a.strip), indent=2))


if __name__ == "__main__":
    main()
