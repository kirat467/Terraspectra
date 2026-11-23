"""Day 25: unit tests + run_all.py + lint + data manifest + README + tag."""
import os, sys, json, glob, hashlib, subprocess
from datetime import date

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Working folder:", os.getcwd())

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest", "ruff", "h5py", "numpy", "scipy"])

os.makedirs("tests", exist_ok=True)
os.makedirs("docs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
# ============================================================
# 1. TESTS
# ============================================================
TESTS = r'''
"""Unit tests with synthetic data (no real data needed). Run: pytest tests -q"""
import os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.cleaning import bad_band_indices, remove_bands, noisy_bands, clip_reflectance, fill_nan_spectral
from src.preprocessing import normalize, smooth_spectra
from src.indices import ndvi
from src.patches import make_patches
from src.split import spatial_block_split
from src.validate import validate_dataset

RNG = np.random.default_rng(0)


def test_bad_band_removal():
    wl = np.arange(400, 2500, 10.0)
    cube = RNG.random((len(wl), 4, 4)).astype("float32")
    bad = bad_band_indices(wl, [[1340, 1470], [1790, 1990]])
    assert len(bad) > 0
    c2, wl2, keep = remove_bands(cube, wl, bad)
    assert c2.shape[0] == len(wl) - len(bad) == len(wl2) == len(keep)
    assert not np.any((wl2 >= 1340) & (wl2 <= 1470))
    assert not np.any((wl2 >= 1790) & (wl2 <= 1990))


def test_noisy_band_detected():
    cube = RNG.normal(0.3, 0.01, (50, 8, 8)).astype("float32")
    cube[10] = RNG.normal(0.3, 1.0, (8, 8))
    assert 10 in noisy_bands(cube)


def test_clip_reflectance_range():
    out = clip_reflectance(np.array([-5.0, 5000.0, 20000.0]), scale=10000)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert abs(out[1] - 0.5) < 1e-6


def test_fill_nan_spectral_interpolates():
    cube = np.tile(np.arange(10, dtype="float32")[:, None, None], (1, 2, 2))
    cube[4, 0, 0] = np.nan
    out = fill_nan_spectral(cube.copy())
    assert not np.isnan(out).any()
    assert abs(out[4, 0, 0] - 4.0) < 1e-5


def test_normalize_per_band_and_minmax():
    cube = (RNG.random((5, 20, 20)) * 3 + 2).astype("float32")
    z = normalize(cube, "per_band")
    assert np.abs(z.mean(axis=(1, 2))).max() < 1e-5
    assert np.abs(z.std(axis=(1, 2)) - 1).max() < 1e-3
    m = normalize(cube, "minmax")
    assert m.min() >= -1e-6 and m.max() <= 1 + 1e-6


def test_smoothing_reduces_noise_and_keeps_nan():
    x = np.sin(np.linspace(0, 6, 100))
    noisy = x + RNG.normal(0, 0.1, 100)
    cube = np.tile(noisy[:, None, None], (1, 3, 3)).astype("float32")
    sm = smooth_spectra(cube, 9, 2)
    assert np.abs(sm[:, 0, 0] - x).mean() < np.abs(noisy - x).mean()
    cube2 = cube.copy()
    cube2[:, 1, 1] = np.nan
    sm2 = smooth_spectra(cube2, 9, 2)
    assert np.isnan(sm2[:, 1, 1]).all() and not np.isnan(sm2[:, 0, 0]).any()


def test_ndvi_value():
    wl = np.array([470.0, 670.0, 800.0])
    cube = np.zeros((3, 2, 2))
    cube[1], cube[2] = 0.1, 0.5
    assert np.allclose(ndvi(cube, wl), 0.4 / 0.6, atol=1e-4)


def test_patches_shapes_and_filters():
    cube = RNG.random((4, 64, 64)).astype("float32")
    labels = np.zeros((64, 64), dtype="int32")
    X, y, coords = make_patches(cube, labels, size=32, stride=32)
    assert X.shape == (4, 4, 32, 32) and (y == 0).all() and len(set(map(tuple, coords.tolist()))) == 4
    c2 = cube.copy()
    c2[:, :32, :32] = np.nan
    assert make_patches(c2, labels, size=32, stride=32)[0].shape[0] == 3
    l2 = labels.copy()
    l2[32:, 32:] = -1
    assert make_patches(cube, l2, size=32, stride=32)[0].shape[0] == 3


def _grid_coords():
    return np.array([(i * 8, j * 8) for i in range(16) for j in range(16)])


def test_spatial_split_no_leakage():
    coords = _grid_coords()
    idx = spatial_block_split(coords, 8, block_size_patches=4, ratios=(0.5, 0.25, 0.25), seed=1)
    allp = np.sort(np.concatenate([idx["train"], idx["val"], idx["test"]]))
    assert np.array_equal(allp, np.arange(len(coords)))
    owner = {}
    for name, ids in idx.items():
        for i in ids:
            r, c = coords[i]
            owner.setdefault((r // 32, c // 32), set()).add(name)
    assert all(len(v) == 1 for v in owner.values()), "a spatial block is shared between splits"


def test_spatial_split_deterministic():
    coords = _grid_coords()
    a = spatial_block_split(coords, 8, 4, (0.7, 0.15, 0.15), seed=42)
    b = spatial_block_split(coords, 8, 4, (0.7, 0.15, 0.15), seed=42)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_validate_dataset_catches_problems():
    X = RNG.random((6, 3, 4, 4)).astype("float32")
    y = np.array([0, 1, 0, 1, 0, 1])
    validate_dataset(X, y, {"train": [0, 1, 2], "val": [3, 4], "test": [5]})
    X2 = X.copy()
    X2[0, 0, 0, 0] = np.nan
    with pytest.raises(AssertionError):
        validate_dataset(X2, y, {"train": [0, 1, 2], "val": [3, 4], "test": [5]})
    with pytest.raises(AssertionError):
        validate_dataset(X, y, {"train": [0, 1], "val": [1, 2], "test": [5]})


H5 = os.path.join(ROOT, "data", "processed", "terraspectra_dataset.h5")


@pytest.mark.skipif(not os.path.exists(H5), reason="final dataset not built yet")
def test_final_dataset_integrity():
    import h5py
    with h5py.File(H5, "r") as f:
        n_classes = int(f.attrs["n_classes"])
        sizes, coords = {}, {}
        for s in ("train", "val", "test"):
            g = f[s]
            y = g["y"][:]
            assert g["X"].shape[0] == len(y)
            if len(y):
                assert y.min() >= 0 and y.max() < n_classes
                assert np.isfinite(g["X"][: min(len(y), 200)]).all()
            sizes[s] = len(y)
            coords[s] = set(map(tuple, g["coords_rc"][:].tolist()))
        assert sizes["train"] > 0
        assert not (coords["train"] & coords["val"])
        assert not (coords["train"] & coords["test"])
        assert not (coords["val"] & coords["test"])
        g = f["train"]
        X, vm = g["X"][:300], g["valid_mask"][:300]
        w = vm[:, None].astype("float64")
        mean = (X * w).sum(axis=(0, 2, 3)) / (w.sum(axis=(0, 2, 3)) + 1e-8)
        assert np.abs(mean).mean() < 0.5, "train data does not look z-score normalized"
'''
open("tests/test_pipeline.py", "w", encoding="utf-8").write(TESTS.lstrip("\n"))
open("tests/__init__.py", "w").write("")
print("Saved tests/test_pipeline.py")
# ============================================================
# 2. run_all.py
# ============================================================
RUN_ALL = r'''
"""Run the whole pipeline in order, with timing and a log.
Usage:
  python scripts/run_all.py                  # processing steps only (no downloads/logins)
  python scripts/run_all.py --with-network   # also the search/download steps
  python scripts/run_all.py --start 5        # begin from step 5
  python scripts/run_all.py --list           # show the steps
"""
import argparse, os, subprocess, sys, time
from datetime import datetime

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEPS = [
    ("scripts/day2_day3_all.py", True, "sources search + Indian Pines test data"),
    ("scripts/day4_5_6_all.py", False, "inventory, metadata, EDA"),
    ("scripts/day21_real_emit_scene.py", True, "real EMIT scene download + ortho crop"),
    ("scripts/day7_8_9_all.py", False, "bad bands, masks, gap fill, reprojection"),
    ("scripts/day10_11_12_all.py", False, "phase 1 summary, normalization, smoothing"),
    ("scripts/day13_14_15_all.py", False, "indices, weather, label alignment"),
    ("scripts/day16_17_18_all.py", False, "patches, spatial split, final dataset"),
    ("scripts/day19_20_all.py", False, "documentation, handover, PPT"),
    ("scripts/day23_baseline_validation.py", False, "baseline validation"),
    ("scripts/day24_final_package.py", False, "report, demo, checklist"),
]

ap = argparse.ArgumentParser()
ap.add_argument("--with-network", action="store_true")
ap.add_argument("--start", type=int, default=1)
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

if a.list:
    for i, (s, net, d) in enumerate(STEPS, 1):
        print(f"{i:2d}. {s}  {'[network]' if net else ''}  {d}")
    sys.exit(0)

os.makedirs("logs", exist_ok=True)
log = open("logs/run_all.log", "a", encoding="utf-8")


def say(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    log.write(line + "\n")
    log.flush()


for i, (script, net, desc) in enumerate(STEPS, 1):
    if i < a.start:
        continue
    if net and not a.with_network:
        say(f"SKIP {i}. {script} (network step; use --with-network)")
        continue
    if not os.path.exists(script):
        say(f"SKIP {i}. {script} (file missing)")
        continue
    say(f"START {i}. {script} - {desc}")
    t0 = time.time()
    rc = subprocess.run([sys.executable, script]).returncode
    say(f"{'DONE ' if rc == 0 else 'FAIL '} {i}. {script} in {time.time() - t0:.0f}s (exit {rc})")
    if rc != 0:
        say("Stopped. Fix the error above and re-run with --start " + str(i))
        sys.exit(rc)
say("ALL STEPS FINISHED")
'''
open("scripts/run_all.py", "w", encoding="utf-8").write(RUN_ALL.lstrip("\n"))
print("Saved scripts/run_all.py")
# ============================================================
# 3. Run tests
# ============================================================
print("\n===== Running tests =====")
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "--tb=short", "-rs"], capture_output=True, text=True)
open("docs/test_report.txt", "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
print("\n".join(r.stdout.strip().splitlines()[-25:]))
tests_ok = r.returncode == 0
print("TESTS:", "ALL PASSED" if tests_ok else "SOME FAILED (see docs/test_report.txt, send me the failure text)")
# ============================================================
# 4. Lint (only real errors: syntax, undefined names, bad comparisons)
# ============================================================
print("\n===== Lint (errors only) =====")
lr = subprocess.run([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests",
                     "--select", "E9,F63,F7,F82", "--output-format", "concise"], capture_output=True, text=True)
open("docs/lint_report.txt", "w", encoding="utf-8").write(lr.stdout + lr.stderr)
n_issues = len([l for l in lr.stdout.splitlines() if ".py:" in l])
print("Lint errors:", n_issues, "(docs/lint_report.txt)")
if n_issues:
    print("\n".join(lr.stdout.splitlines()[:10]))
# ============================================================
# 5. Data manifest (hashes of final files)
# ============================================================


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = []
for pat in ("data/processed/*.h5", "data/processed/*.json", "data/processed/*.npz", "data/processed/*.npy",
            "data/metadata/*.json", "data/metadata/*.csv", "outputs/*.zip"):
    files += glob.glob(pat)
manifest = {"created": str(date.today()),
            "files": {p.replace("\\", "/"): {"bytes": os.path.getsize(p), "sha256": sha(p)} for p in sorted(set(files))
                      if os.path.getsize(p) < 3_000_000_000}}
json.dump(manifest, open("docs/data_manifest.json", "w"), indent=2)
print(f"\nSaved docs/data_manifest.json ({len(manifest['files'])} files hashed)")
# ============================================================
# 6. README section
# ============================================================
marker = "## Testing and full re-run"
rd = open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else ""
if marker not in rd:
    with open("README.md", "a", encoding="utf-8") as f:
        f.write(f"""

{marker}
```bash
pip install -r requirements.txt
pytest tests -q                              # unit tests (synthetic data, no downloads)
python scripts/run_all.py --list             # show pipeline steps
python scripts/run_all.py                    # processing steps only
python scripts/run_all.py --with-network     # also search/download steps (needs Earthdata login)
streamlit run scripts/demo_app.py            # interactive demo
```
- Test report: docs/test_report.txt | Lint report: docs/lint_report.txt
- File hashes of final outputs: docs/data_manifest.json
- All randomness uses fixed seeds (42); reproducibility check: docs/reproducibility_check.md
""")
    print("README.md updated")
    # ============================================================
# 7. Git
# ============================================================
try:
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-m", "Day 25: unit tests, run_all, lint, data manifest"], check=False)
    if tests_ok:
        subprocess.run(["git", "tag", "-f", "v1.1-final"], check=True)
    if subprocess.run(["git", "remote"], capture_output=True, text=True).stdout.strip():
        subprocess.run(["git", "push"], check=False)
        if tests_ok:
            subprocess.run(["git", "push", "-f", "origin", "v1.1-final"], check=False)
except Exception as e:
    print("Git issue:", e)

print("\nDAY 25 DONE." + (" Tag v1.1-final created." if tests_ok else " Tag NOT created because tests failed."))