import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_large_scene as ml  # noqa: E402
import memory_audit as ma  # noqa: E402


def small_cube(tmp_path):
    ml.generate(tmp_path, size=64, step=3, strip=32)
    return tmp_path / "large_cube_3.npy"


def test_projection_same_for_any_strip(tmp_path):
    cube = np.load(small_cube(tmp_path), mmap_mode="r")
    mean, axes = ma.fit_pca(cube, n_rows=16)
    a = ma.project_in_strips(cube, mean, axes, 8)
    b = ma.project_in_strips(cube, mean, axes, 64)
    assert a.shape == (64, 64, 20) and np.allclose(a, b, atol=1e-5)


def test_pca_axes_are_orthonormal(tmp_path):
    cube = np.load(small_cube(tmp_path), mmap_mode="r")
    _, axes = ma.fit_pca(cube, n_rows=16)
    assert axes.shape == (200, 20) and np.allclose(axes.T @ axes, np.eye(20), atol=1e-3)


def test_audit_runs_and_reports_every_setting(tmp_path):
    info = ma.audit(small_cube(tmp_path), quick=True)
    assert {r["setting"] for r in info["load"]} == {"full", "mmap"}
    assert len(info["project"]) == 2 and len(info["train"]) == 3
    assert all(r["peak_mb"] > 0 and r["heap_mb"] >= 0 for k in ("load", "project", "train") for r in info[k])
    ma.chart(info, tmp_path / "chart.png")
    assert (tmp_path / "chart.png").stat().st_size > 5000
