import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_large_scene as ml  # noqa: E402


def test_strip_size_does_not_change_output(tmp_path):
    a = ml.generate(tmp_path / "a", size=96, step=3, strip=16)
    b = ml.generate(tmp_path / "b", size=96, step=3, strip=48)
    ca = np.load(tmp_path / "a" / "large_cube_3.npy"); cb = np.load(tmp_path / "b" / "large_cube_3.npy")
    assert np.array_equal(ca, cb) and a["class_pixels"] == b["class_pixels"]


def test_shape_dtype_and_range(tmp_path):
    ml.generate(tmp_path, size=80, step=2, strip=32)
    cube = np.load(tmp_path / "large_cube_2.npy", mmap_mode="r")
    assert cube.shape == (80, 80, 200) and cube.dtype == np.uint16 and cube.max() <= 10000


def test_outbreaks_grow_with_step(tmp_path):
    counts = []
    for s in (0, 2, 5):
        r = ml.generate(tmp_path / str(s), size=128, step=s, strip=64)
        counts.append(r["class_pixels"]["early_stress"] + r["class_pixels"]["visible_stress"])
    assert counts[0] < counts[1] < counts[2]


def test_labels_match_layout(tmp_path):
    ml.generate(tmp_path, size=128, step=5, strip=64)
    lab = np.load(tmp_path / "large_labels_5.npy")
    crop, _ = ml.layout(128)
    assert (lab[crop < 0] == -1).all() and set(np.unique(lab)) <= {-1, 0, 1, 2}
    assert (lab == 2).any() and (lab == 0).any()
