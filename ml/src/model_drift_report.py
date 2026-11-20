import math


def generate_drift_report(drift_result):
    if not isinstance(drift_result, dict):
        raise ValueError("Drift result must be a dictionary")

    required = [
        "drift_detected",
        "distribution_drift",
        "drift_threshold",
        "reference_distribution",
        "current_distribution"
    ]

    for key in required:
        if key not in drift_result:
            raise ValueError(f"Missing field: {key}")

    detected = drift_result["drift_detected"]
    score = drift_result["distribution_drift"]
    threshold = drift_result["drift_threshold"]

    if not isinstance(detected, bool):
        raise ValueError("Drift status must be boolean")

    for value in [score, threshold]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Drift scores must be numeric")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Drift scores must be between 0 and 1")

    reference = drift_result["reference_distribution"]
    current = drift_result["current_distribution"]

    if not isinstance(reference, (list, tuple)):
        raise ValueError("Reference distribution must be a list")

    if not isinstance(current, (list, tuple)):
        raise ValueError("Current distribution must be a list")

    if len(reference) == 0 or len(reference) != len(current):
        raise ValueError("Distributions must have matching non-empty lengths")

    changes = {}

    for i, (old, new) in enumerate(zip(reference, current)):
        for value in [old, new]:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Distribution values must be numeric")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Distribution values must be between 0 and 1")

        changes[i] = round(new - old, 4)

    report = {
        "status": "Drift Detected" if detected else "No Significant Drift",
        "distribution_drift": round(score, 4),
        "drift_threshold": round(threshold, 4),
        "class_changes": changes,
        "recommendation": (
            "Review predictions and investigate possible data changes."
            if detected
            else "Continue monitoring model predictions."
        )
    }

    confidence_change = drift_result.get("confidence_change")

    if confidence_change is not None:
        if (
            isinstance(confidence_change, bool)
            or not isinstance(confidence_change, (int, float))
            or not math.isfinite(confidence_change)
        ):
            raise ValueError("Confidence change must be a finite number")

        report["confidence_change"] = round(confidence_change, 4)

    return report
