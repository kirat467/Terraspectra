import numpy as np
import pytest

from risk_map import (build_payload, components, disease_probability, forecast_risk, resample_mean, run,
                      scan_disease_map)


def fake_predict(patches):
    """Pretend model: a patch is diseased when its mean reflectance is high."""
    return np.clip(patches.mean(axis=(1, 2, 3, 4)), 0, 1)


def make_cube(size=48, bands=6, spot=None):
    cube = np.full((size, size, bands), 0.1, dtype=np.float32)
    if spot:
        r, c, rad = spot
        yy, xx = np.mgrid[0:size, 0:size]
        cube[(yy - r) ** 2 + (xx - c) ** 2 <= rad ** 2] = 0.95
    return cube


def test_disease_probability_is_one_minus_healthy():
    p = disease_probability([[0.9, 0.1], [0.2, 0.8]])
    assert p == pytest.approx([0.1, 0.8])


def test_scan_map_finds_the_spot_and_covers_every_pixel():
    m = scan_disease_map(make_cube(spot=(12, 12, 8)), fake_predict, patch_size=8, stride=4)
    assert m.shape == (48, 48)
    assert np.isfinite(m).all()
    assert m[12, 12] > 0.7 and m[40, 40] < 0.15


def test_scan_map_rejects_small_or_bad_cubes():
    with pytest.raises(ValueError):
        scan_disease_map(np.zeros((4, 4, 3)), fake_predict, patch_size=8)
    bad = make_cube(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        scan_disease_map(bad, fake_predict, patch_size=8)


def test_resample_keeps_the_average():
    a = np.random.default_rng(0).random((48, 48))
    assert resample_mean(a, 12, 12).mean() == pytest.approx(a.mean())
    assert resample_mean(a, 48, 48).shape == (48, 48)


def test_forecast_adds_risk_only_where_disease_is_growing():
    grow = np.array([[0.1, 0.1]]); 
    maps = [np.array([[0.1, 0.5]]), np.array([[0.2, 0.5]]), np.array([[0.3, 0.5]])]
    out = forecast_risk(maps, ["2026-10-06", "2026-10-13", "2026-10-20"])
    assert out[0, 0] > 0.3 + 0.1          # growing pixel gets extra risk
    assert out[0, 1] == pytest.approx(0.5)  # flat pixel keeps its level
    falling = forecast_risk([np.array([[0.6]]), np.array([[0.4]])], ["2026-10-06", "2026-10-13"])
    assert falling[0, 0] == pytest.approx(0.4)


def test_forecast_single_scan_and_validation():
    assert forecast_risk([np.array([[0.4]])], ["2026-10-06"])[0, 0] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        forecast_risk([np.zeros((1, 1))], ["2026-10-06", "2026-10-13"])


def test_components_ignores_tiny_groups():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:5, 1:5] = True
    mask[8, 8] = True
    assert [len(g) for g in components(mask)] == [16]


def farm(size=20):
    west, south, east, north = 76.0, 12.0, 76.01, 12.01
    def ring(x0, y0, x1, y1):
        f = lambda x, y: [west + x / size * (east - west), north - y / size * (north - south)]
        return [f(x0, y0), f(x1, y0), f(x1, y1), f(x0, y1), f(x0, y0)]
    geo = {"type": "FeatureCollection", "features": [
        {"properties": {"name": "Field A1", "crop": "Paddy"}, "geometry": {"type": "Polygon", "coordinates": [ring(1, 1, 9, 19)]}},
        {"properties": {"name": "Field A2", "crop": "Maize"}, "geometry": {"type": "Polygon", "coordinates": [ring(11, 1, 19, 19)]}}]}
    meta = {"bounds": [west, south, east, north], "grid": [size, size], "pixel_size_m": 15.0, "simulated": True}
    return meta, geo


def test_payload_matches_the_dashboard_format():
    meta, geo = farm()
    risk = np.full((20, 20), 0.05); risk[4:10, 2:8] = 0.9
    p = build_payload(risk, "2026-11-10", meta, geo)
    assert p["rows"] == p["cols"] == 20 and p["horizon_days"] == 21
    assert p["risk"][0][0] is None and p["risk"][5][3] == 0.9       # track vs crop cell
    assert p["summary"]["zones"] == 1 and p["zones"][0]["field"] == "Field A1" and p["zones"][0]["id"] == "Z1"
    acres = 36 * 15 ** 2 / 4046.8564
    assert p["summary"]["acres_high"] == pytest.approx(acres, abs=0.06)
    assert {f["name"] for f in p["fields"]} == {"Field A1", "Field A2"}
    json_ready = __import__("json").dumps(p)       # no NaN left over
    assert "NaN" not in json_ready


def test_run_writes_a_timeline_the_dashboard_can_load(tmp_path):
    meta, geo = farm()
    scans = [("2026-10-06", make_cube(48, spot=(14, 14, 3))), ("2026-10-13", make_cube(48, spot=(14, 14, 6))),
             ("2026-10-20", make_cube(48, spot=(14, 14, 9)))]
    index = run(scans, fake_predict, meta, geo, tmp_path, patch_size=8, stride=4)
    assert index["dates"] == ["2026-10-06", "2026-10-13", "2026-10-20"] and index["latest"] == "2026-10-20"
    assert (tmp_path / "risk_2026-10-13.json").exists()
    acres = [h["acres_high"] for h in index["history"]]
    assert acres[-1] >= acres[0]
