import pytest

from model_drift import detect_model_drift


def test_no_drift():
    result = detect_model_drift(
        [0, 0, 1, 1],
        [0, 0, 1, 1]
    )

    assert result["distribution_drift"] == 0.0
    assert result["drift_detected"] is False
    assert result["status"] == "No Significant Drift"


def test_drift_detected():
    result = detect_model_drift(
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        drift_threshold=0.1
    )

    assert result["distribution_drift"] == 0.75
    assert result["drift_detected"] is True
    assert result["status"] == "Drift Detected"


def test_reference_distribution():
    result = detect_model_drift(
        [0, 0, 1, 1],
        [0, 1, 1, 1]
    )

    assert result["reference_distribution"] == [0.5, 0.5]
    assert result["current_distribution"] == [0.25, 0.75]


def test_confidence_change():
    result = detect_model_drift(
        [0, 1],
        [0, 1],
        reference_confidences=[0.8, 0.9],
        current_confidences=[0.5, 0.6]
    )

    assert result["confidence_change"] == pytest.approx(-0.3)


def test_sample_counts():
    result = detect_model_drift(
        [0, 1, 1],
        [0, 0]
    )

    assert result["reference_samples"] == 3
    assert result["current_samples"] == 2


def test_empty_predictions():
    with pytest.raises(ValueError):
        detect_model_drift([], [0, 1])


def test_invalid_prediction_dimensions():
    with pytest.raises(ValueError):
        detect_model_drift([[0, 1]], [0, 1])


def test_invalid_confidence_range():
    with pytest.raises(ValueError):
        detect_model_drift(
            [0, 1],
            [0, 1],
            reference_confidences=[1.2, 0.8],
            current_confidences=[0.7, 0.9]
        )


def test_invalid_drift_threshold():
    with pytest.raises(ValueError):
        detect_model_drift(
            [0, 1],
            [0, 1],
            drift_threshold=1.5
        )


def test_unknown_class_label():
    with pytest.raises(ValueError):
        detect_model_drift(
            [0, 1],
            [0, 2],
            num_classes=2
        )


def test_confidence_length_mismatch():
    with pytest.raises(ValueError):
        detect_model_drift(
            [0, 1],
            [0, 1],
            reference_confidences=[0.8],
            current_confidences=[0.7, 0.9]
        )


def test_missing_confidence_array():
    with pytest.raises(ValueError):
        detect_model_drift(
            [0, 1],
            [0, 1],
            reference_confidences=[0.8, 0.9]
        )