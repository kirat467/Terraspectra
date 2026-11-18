import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))
pytest.importorskip("rasterio")
pytest.importorskip("h5py")
from check_alignment import check, lonlat_to_pixel  # noqa: E402
from load_scene import list_dates, load_scene  # noqa: E402
from make_scene import build_scenes, write_geotiff, write_hdf5  # noqa: E402

FARM = ROOT / "frontend" / "public" / "data"


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("scenes")
    meta, geo, owner, sc = build_scenes(FARM, dates=["2026-10-06", "2026-10-13"])
    write_hdf5(d / "s.h5", sc, meta)
    write_geotiff(d / "a.tif", sc[1]["cube"], meta["bounds"], sc[1]["date"])
    return d, sc, meta


def test_h5_round_trip(files):
    d, sc, meta = files
    assert list_dates(d / "s.h5") == ["2026-10-06", "2026-10-13"]
    s = load_scene(d / "s.h5", "2026-10-13")
    assert s["cube"].shape == (135, 135, 200) and s["cube"].dtype == np.float32
    assert np.abs(s["cube"] - sc[1]["cube"]).max() < 1e-4
    assert (s["label"] == sc[1]["label"]).all() and s["wavelengths_nm"][0] == 400 and s["simulated"]
    assert load_scene(d / "s.h5")["date"] == "2026-10-13"          # default is the latest scan
    with pytest.raises(KeyError):
        load_scene(d / "s.h5", "2030-01-01")


def test_tif_round_trip(files):
    d, sc, meta = files
    s = load_scene(d / "a.tif")
    assert s["date"] == "2026-10-13" and s["wavelengths_nm"].shape == (200,)
    assert np.abs(s["cube"] - sc[1]["cube"]).max() < 1e-4
    assert s["bounds"] == pytest.approx(meta["bounds"])
    with pytest.raises(ValueError):
        load_scene(d / "a.txt")


def test_alignment_passes_for_the_generated_scene(files):
    d, _, _ = files
    rows = check(d / "a.tif", FARM)
    assert all(r[3] for r in rows), [r for r in rows if not r[3]]


def test_alignment_catches_a_shifted_scene(files, tmp_path):
    d, sc, meta = files
    west, south, east, north = meta["bounds"]
    shifted = [west + 0.0003, south, east + 0.0003, north]        # about 3 pixels east
    write_geotiff(tmp_path / "bad.tif", sc[1]["cube"], shifted, "2026-10-13")
    bad = [r[0] for r in check(tmp_path / "bad.tif", FARM) if not r[3]]
    assert "west edge" in bad and "east edge" in bad


def test_pixel_maths_on_known_corners():
    b = [10.0, 0.0, 11.0, 1.0]
    assert lonlat_to_pixel(10.0, 1.0, b, 100, 100) == (0.0, 0.0)
    assert lonlat_to_pixel(11.0, 0.0, b, 100, 100) == (100.0, 100.0)
    assert lonlat_to_pixel(10.5, 0.5, b, 100, 100) == (50.0, 50.0)
