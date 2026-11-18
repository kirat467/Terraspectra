import numpy as np
import pytest

from train_on_scenes import oversample, parse_history, per_class_report, standardise, to_cnn


def test_standardise_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    train = rng.normal(5, 2, size=(50, 3, 3, 4)).astype(np.float32)
    test = rng.normal(100, 2, size=(10, 3, 3, 4)).astype(np.float32)
    (mean, std), (tr, te) = standardise(train, test)
    assert np.allclose(tr.mean(axis=(0, 1, 2)), 0, atol=1e-5) and np.allclose(tr.std(axis=(0, 1, 2)), 1, atol=1e-4)
    assert te.mean() > 10                                  # test is scaled with train numbers, so a shift stays visible
    assert mean.shape == (1, 1, 1, 4)


def test_oversample_only_repeats_rare_classes():
    X = np.arange(100, dtype=np.float32).reshape(100, 1, 1, 1)
    y = np.array([0] * 90 + [1] * 8 + [2] * 2)
    X2, y2 = oversample(X, y, target_ratio=0.5)
    counts = np.bincount(y2, minlength=3)
    assert counts[0] == 90 and counts[1] == 45 and counts[2] == 45
    assert set(X2[y2 == 2].ravel()) <= set(X[y == 2].ravel())       # copies of real patches, nothing invented


def test_oversample_skips_a_class_with_no_patches():
    y = np.array([0] * 20 + [1] * 2)
    _, y2 = oversample(np.zeros((22, 1, 1, 1), np.float32), y, 0.5)
    assert np.bincount(y2, minlength=3)[2] == 0


def test_cnn_layout():
    assert to_cnn(np.zeros((7, 9, 9, 20), np.float32)).shape == (7, 1, 20, 9, 9)


def test_parse_history_reads_the_training_log():
    log = "Epoch 1/5 | Training Loss: 0.2500 | Training Accuracy: 90.00%\nValidation Loss: 0.0200 | Validation Accuracy: 99.00%\nEpoch 2/5 | Training Loss: 0.0400 | x\nValidation Loss: 0.0190 | y"
    assert parse_history(log) == ([0.25, 0.04], [0.02, 0.019])


def test_per_class_report_counts_and_zero_division():
    y = np.array([0, 0, 0, 1, 1, 2])
    pred = np.array([0, 0, 1, 1, 1, 1])
    r = per_class_report(y, pred)
    assert r["confusion_matrix"] == [[2, 1, 0], [0, 2, 0], [0, 1, 0]]
    assert r["per_class"]["visible_stress"]["recall"] == 0.0 and r["per_class"]["early_stress"]["recall"] == 1.0
    assert r["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
