import numpy as np


def detect_model_drift(
    reference_predictions,
    current_predictions,
    reference_confidences=None,
    current_confidences=None,
    num_classes=None,
    drift_threshold=0.1
):
    reference = np.asarray(reference_predictions)
    current = np.asarray(current_predictions)

    if reference.ndim != 1 or current.ndim != 1:
        raise ValueError("Predictions must be one-dimensional")

    if len(reference) == 0 or len(current) == 0:
        raise ValueError("Prediction arrays cannot be empty")

    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(current)):
        raise ValueError("Predictions must contain finite values")

    if not np.all(reference == reference.astype(int)):
        raise ValueError("Predictions must be integer class labels")

    if not np.all(current == current.astype(int)):
        raise ValueError("Predictions must be integer class labels")

    reference = reference.astype(int)
    current = current.astype(int)

    if np.any(reference < 0) or np.any(current < 0):
        raise ValueError("Class labels cannot be negative")

    if (
        not isinstance(drift_threshold, (int, float))
        or isinstance(drift_threshold, bool)
        or not np.isfinite(drift_threshold)
        or not 0 <= drift_threshold <= 1
    ):
        raise ValueError("Drift threshold must be between 0 and 1")

    if num_classes is None:
        num_classes = int(max(reference.max(), current.max()) + 1)

    if (
        not isinstance(num_classes, (int, np.integer))
        or isinstance(num_classes, (bool, np.bool_))
        or num_classes <= 0
    ):
        raise ValueError("Number of classes must be a positive integer")

    num_classes = int(num_classes)

    if np.any(reference >= num_classes) or np.any(current >= num_classes):
        raise ValueError("Prediction contains an unknown class label")

    reference_distribution = (
        np.bincount(reference, minlength=num_classes) / len(reference)
    )

    current_distribution = (
        np.bincount(current, minlength=num_classes) / len(current)
    )

    class_changes = np.abs(
        current_distribution - reference_distribution
    )

    distribution_drift = float(np.sum(class_changes) / 2)

    confidence_change = None

    if (reference_confidences is None) != (current_confidences is None):
        raise ValueError("Provide both confidence arrays or neither")

    if reference_confidences is not None:
        ref_conf = np.asarray(reference_confidences, dtype=float)
        cur_conf = np.asarray(current_confidences, dtype=float)

        if ref_conf.ndim != 1 or cur_conf.ndim != 1:
            raise ValueError("Confidence arrays must be one-dimensional")

        if len(ref_conf) != len(reference) or len(cur_conf) != len(current):
            raise ValueError("Confidence and prediction lengths must match")

        if not np.all(np.isfinite(ref_conf)) or not np.all(np.isfinite(cur_conf)):
            raise ValueError("Confidences must contain finite values")

        if (
            np.any((ref_conf < 0) | (ref_conf > 1))
            or np.any((cur_conf < 0) | (cur_conf > 1))
        ):
            raise ValueError("Confidences must be between 0 and 1")

        confidence_change = float(
            np.mean(cur_conf) - np.mean(ref_conf)
        )

    return {
        "reference_samples": len(reference),
        "current_samples": len(current),
        "reference_distribution": reference_distribution.tolist(),
        "current_distribution": current_distribution.tolist(),
        "class_distribution_changes": class_changes.tolist(),
        "distribution_drift": distribution_drift,
        "confidence_change": confidence_change,
        "drift_detected": distribution_drift > drift_threshold,
        "drift_threshold": float(drift_threshold),
        "status": (
            "Drift Detected"
            if distribution_drift > drift_threshold
            else "No Significant Drift"
        )
    }