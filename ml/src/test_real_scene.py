"""Tests for the real-scene adapter (scripts/real_scene_to_dashboard.py) using the fake EMIT stand-in. Run: pytest ml/src/test_real_scene.py"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ml" / "src"))
import make_fake_emit as mf  # noqa: E402
import real_scene_to_dashboard as rs  # noqa: E402


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("emit")
    cube, wl, good, meta = mf.make_fake(96)
    np.save(d / "real_scene_cube.npy", cube); np.save(d / "real_scene_wl.npy", wl)
    np.save(d / "real_scene_good.npy", good); (d / "real_scene_meta.json").write_text(json.dumps(meta))
    return d


def test_stress_is_highest_in_planted_patch(scene):
    cube, wl, good, _ = rs.load_scene(scene)
    refl, wl2, usable = rs.prepare(cube, wl, good)
    risk, veg, _ = rs.stress_map(refl, wl2, usable)
    yy, xx = np.mgrid[0:96, 0:96]
    patch = ((xx - 0.72 * 96) ** 2 + (yy - 0.28 * 96) ** 2) < (0.08 * 96) ** 2
    far = veg & (xx > 0.15 * 96) & (yy > 0.65 * 96)
    assert np.nanmean(risk[patch & veg]) > 0.4
    assert np.nanmean(risk[far]) < 0.1


def test_masked_soil_water_and_cloud_are_excluded(scene):
    cube, wl, good, _ = rs.load_scene(scene)
    refl, wl2, usable = rs.prepare(cube, wl, good)
    risk, veg, _ = rs.stress_map(refl, wl2, usable)
    assert not veg[:, :3].any()                  # NaN edge
    assert not veg[:, :6].any()                  # bare strip
    assert np.isnan(risk[~veg]).all()


def test_bad_bands_dropped(scene):
    cube, wl, good, _ = rs.load_scene(scene)
    refl, wl2, _ = rs.prepare(cube, wl, good)
    assert refl.shape[2] == good.sum() == len(wl2)


def test_outputs_match_dashboard_contract(scene, tmp_path):
    info = rs.build_dashboard_files(scene, tmp_path)
    meta = json.loads((tmp_path / "farm_meta.json").read_text())
    assert meta["simulated"] is True and "Stand-in" in meta["source"]      # fake data must be labelled
    idx = json.loads((tmp_path / "risk" / "risk_index.json").read_text())
    p = json.loads((tmp_path / "risk" / f"risk_{idx['dates'][0]}.json").read_text())
    assert p["kind"] == "stress_index" and len(p["risk"]) == p["rows"] and len(p["risk"][0]) == p["cols"]
    assert (tmp_path / "farm_rgb.png").exists() and info["vegetated_pixels"] > 1000
    assert json.loads((tmp_path / "farm_fields.geojson").read_text())["features"] == []


def test_too_little_vegetation_raises():
    refl = np.full((20, 20, 10), 0.3, dtype=np.float32)
    wl = np.linspace(400, 2400, 10)
    with pytest.raises(ValueError):
        rs.stress_map(refl, wl, np.ones((20, 20), bool))
