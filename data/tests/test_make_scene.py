import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
from make_scene import SCAN_DATES, build_scenes, write_geotiff, write_hdf5  # noqa: E402

FARM = ROOT / "frontend" / "public" / "data"


@pytest.fixture(scope="module")
def scenes():
    return build_scenes(FARM, dates=SCAN_DATES)


def test_cubes_match_the_dashboard_grid(scenes):
    meta, geo, owner, sc = scenes
    assert len(sc) == 6 and [s["date"] for s in sc] == SCAN_DATES
    for s in sc:
        assert s["cube"].shape == (135, 135, 200) and s["cube"].dtype == np.float32
        assert np.isfinite(s["cube"]).all() and 0 <= s["cube"].min() and s["cube"].max() <= 1


def test_disease_grows_from_an_early_core(scenes):
    _, _, _, sc = scenes
    early = [int((s["label"] == 1).sum()) for s in sc]
    visible = [int((s["label"] == 2).sum()) for s in sc]
    assert early[0] > 0 and visible[0] == 0                  # the first scan shows only an early-stress core
    assert early == sorted(early) and visible == sorted(visible)
    assert visible[-1] > 100


def test_tracks_and_fallow_are_not_labelled_as_crop(scenes):
    meta, geo, owner, sc = scenes
    label = sc[-1]["label"]
    assert (label[owner < 0] == -1).all()
    fallow = [k for k, f in enumerate(geo["features"]) if f["properties"]["crop"] == "Fallow"]
    assert fallow and (label[np.isin(owner, fallow)] == -1).all()


def test_outbreak_sits_in_field_b2(scenes):
    meta, geo, owner, sc = scenes
    names = [f["properties"]["name"] for f in geo["features"]]
    stressed = sc[-1]["label"] >= 1
    counts = np.bincount(owner[stressed & (owner >= 0)], minlength=len(names))
    assert names[int(counts.argmax())] == "Field B2"


def test_geotiff_and_hdf5_round_trip(scenes, tmp_path):
    rasterio = pytest.importorskip("rasterio")
    h5py = pytest.importorskip("h5py")
    meta, geo, owner, sc = scenes
    write_geotiff(tmp_path / "a.tif", sc[-1]["cube"], meta["bounds"], sc[-1]["date"])
    with rasterio.open(tmp_path / "a.tif") as src:
        assert (src.count, src.height, src.width) == (200, 135, 135)
        assert src.crs.to_string() == "EPSG:4326"
        west, south, east, north = meta["bounds"]
        assert src.bounds.left == pytest.approx(west) and src.bounds.top == pytest.approx(north)
        assert src.descriptions[0] == "400 nm" and src.descriptions[-1] == "2400 nm"
        back = src.read().transpose(1, 2, 0) / 10000
    assert np.abs(back - sc[-1]["cube"]).max() < 1e-4
    write_hdf5(tmp_path / "s.h5", sc[:2], meta)
    with h5py.File(tmp_path / "s.h5") as h:
        assert h["cube/2026-10-06"].shape == (135, 135, 200) and h["wavelengths_nm"].shape == (200,)
        assert h.attrs["simulated"]
