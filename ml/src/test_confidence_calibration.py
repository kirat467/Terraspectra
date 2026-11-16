import pytest

from confidence_calibration import calculate_confidence_calibration


def test_calibration_basic():
    result = calculate_confidence_calibration(
        [0.9, 0.8, 0.7, 0.6],
        [1, 0, 1, 0],
        [1, 0, 1, 0]
    )

    assert result["total_samples"] == 4
    assert result["accuracy"] == 1.0
    assert result["reliability"] == "Underconfident"


def test_perfect_confidence():
    result = calculate_confidence_calibration(
        [1.0, 1.0],
        [1, 0],
        [1, 0]
    )

    assert result["accuracy"] == 1.0
    assert result["expected_calibration_error"] == 0.0
    assert result["brier_score"] == 0.0


def test_overconfident_predictions():
    result = calculate_confidence_calibration(
        [0.9, 0.9],
        [1, 1],
        [0, 0]
    )

    assert result["reliability"] == "Overconfident"
    assert result["accuracy"] == 0.0


def test_underconfident_predictions():
    result = calculate_confidence_calibration(
        [0.3, 0.3],
        [1, 0],
        [1, 0]
    )

    assert result["reliability"] == "Underconfident"


def test_calibration_bins():
    result = calculate_confidence_calibration(
        [0.2, 0.8],
        [1, 0],
        [1, 0],
        num_bins=5
    )

    assert len(result["calibration_bins"]) == 2


def test_empty_inputs():
    with pytest.raises(ValueError):
        calculate_confidence_calibration([], [], [])


def test_different_input_lengths():
    with pytest.raises(ValueError):
        calculate_confidence_calibration(
            [0.8, 0.9],
            [1],
            [1, 0]
        )


def test_invalid_confidence():
    with pytest.raises(ValueError):
        calculate_confidence_calibration(
            [1.5],
            [1],
            [1]
        )


def test_invalid_number_of_bins():
    with pytest.raises(ValueError):
        calculate_confidence_calibration(
            [0.8],
            [1],
            [1],
            num_bins=0
        )


def test_non_finite_confidence():
    with pytest.raises(ValueError):
        calculate_confidence_calibration(
            [float("nan")],
            [1],
            [1]
        )