import pytest

from model_drift_report import generate_drift_report


@pytest.fixture
def sample_result():
    return {
        "reference_samples": 100,
        "current_samples": 100,
        "reference_distribution": [0.5, 0.3, 0.2],
        "current_distribution": [0.3, 0.4, 0.3],
        "class_distribution_changes": [0.2, 0.1, 0.1],
        "distribution_drift": 0.2,
        "confidence_change": -0.05,
        "drift_detected": True,
        "drift_threshold": 0.1,
        "status": "Drift Detected"
    }


def test_generate_report(sample_result):
    report = generate_drift_report(sample_result)

    assert report["status"] == "Drift Detected"
    assert report["distribution_drift"] == 0.2
    assert report["class_changes"] == {
        0: -0.2,
        1: 0.1,
        2: 0.1
    }
    assert report["confidence_change"] == -0.05


def test_no_drift(sample_result):
    sample_result["drift_detected"] = False

    report = generate_drift_report(sample_result)

    assert report["status"] == "No Significant Drift"
    assert "Continue monitoring" in report["recommendation"]


def test_missing_field(sample_result):
    del sample_result["distribution_drift"]

    with pytest.raises(ValueError):
        generate_drift_report(sample_result)


def test_invalid_input():
    with pytest.raises(ValueError):
        generate_drift_report([])


def test_mismatched_distributions(sample_result):
    sample_result["current_distribution"] = [0.5, 0.5]

    with pytest.raises(ValueError):
        generate_drift_report(sample_result)


def test_invalid_distribution_values(sample_result):
    sample_result["reference_distribution"] = [0.5, 1.5, 0.2]

    with pytest.raises(ValueError):
        generate_drift_report(sample_result)


def test_invalid_drift_score(sample_result):
    sample_result["distribution_drift"] = 1.5

    with pytest.raises(ValueError):
        generate_drift_report(sample_result)


def test_confidence_not_present(sample_result):
    sample_result["confidence_change"] = None

    report = generate_drift_report(sample_result)

    assert "confidence_change" not in report