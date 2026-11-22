"""Tests for the lead-time maths (synthetic stacks, no model needed). Run: pytest ml/src/test_lead_time.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lead_time as lt  # noqa: E402

DAYS = [0, 7, 14, 21, 28, 35]


def stack(first_flag, first_vis):
    """Left 4 columns: stressed pixels (score jumps to 1 at scan first_flag, label becomes 2 at scan first_vis).
    Right 4 columns: healthy pixels with scores below 0.5."""
    score = np.zeros((6, 4, 8)); label = np.zeros((6, 4, 8), dtype=np.int8)
    score[:, :, 4:] = np.random.default_rng(0).uniform(0, 0.5, size=(6, 4, 4))
    if first_flag is not None:
        score[first_flag:, :, :4] = 1.0
    label[first_vis:, :, :4] = 2
    return score, label


def test_first_true_and_never():
    h = np.zeros((6, 2, 2), dtype=bool); h[3:, 0, 0] = True
    f = lt.first_true(h)
    assert f[0, 0] == 3 and f[1, 1] == -1


def test_lead_is_days_between_flag_and_visible():
    score, label = stack(1, 3)                    # flagged on scan 1 (day 7), visible on scan 3 (day 21)
    r = lt.evaluate(score, label, DAYS, np.ones((4, 8), bool), np.ones((4, 8), bool))
    assert r["median_lead_days"] == 14 and r["shares"]["14 days before"] == 1.0 and r["never_flagged"] == 0


def test_flag_after_visible_is_negative_and_missed_is_counted():
    score, label = stack(4, 2)
    assert lt.evaluate(score, label, DAYS, np.ones((4, 8), bool), np.ones((4, 8), bool))["shares"]["after visible"] == 1.0
    score, label = stack(None, 2)
    assert lt.evaluate(score, label, DAYS, np.ones((4, 8), bool), np.ones((4, 8), bool))["never_flagged"] == 1.0


def test_first_scan_flags_are_marked_as_lower_bound():
    score, label = stack(0, 4)
    r = lt.evaluate(score, label, DAYS, np.ones((4, 8), bool), np.ones((4, 8), bool))
    assert r["flagged_on_first_scan"] == 1.0 and r["shares"]["21+ days before"] == 1.0


def test_threshold_uses_only_healthy_pixels_and_sets_false_alarm_rate():
    rng = np.random.default_rng(0)
    healthy = rng.normal(size=100000)
    thr = lt.threshold_for_fpr(healthy, 0.01)
    assert (healthy >= thr).mean() == pytest.approx(0.01, abs=0.002)


def test_ground_truth_head_start():
    label = np.zeros((6, 3, 3), dtype=np.int8); label[1:, 0, 0] = 1; label[3:, 0, 0] = 2
    g = lt.ground_truth_lead(label, DAYS, np.ones((3, 3), bool))
    assert g["median_lead_days"] == 14 and g["pixels"] == 1


def test_box_mean_keeps_shape_and_averages():
    a = np.zeros((2, 7, 7)); a[:, 3, 3] = 25.0
    b = lt.box_mean(a, 5)
    assert b.shape == a.shape and b[0, 3, 3] == pytest.approx(1.0)
