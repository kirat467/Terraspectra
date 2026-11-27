import numpy as np


def _validate_numeric_array(values, name):
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain valid numeric values") from error

    if array.ndim != 1:
        raise ValueError(f"{name} must be 1D")

    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric values")

    if np.issubdtype(array.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real values")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")

    return array


def calculate_confidence_calibration(
    confidences,
    predictions,
    actual_labels,
    num_bins=10
):
    if isinstance(confidences, (bool, np.bool_)):
        raise ValueError("Confidences must be numeric")

    if isinstance(predictions, (bool, np.bool_)):
        raise ValueError("Predictions must contain integer class labels")

    if isinstance(actual_labels, (bool, np.bool_)):
        raise ValueError("Actual labels must contain integer class labels")

    if isinstance(num_bins, (bool, np.bool_)) or not isinstance(
        num_bins, (int, np.integer)
    ):
        raise ValueError("Number of bins must be a positive integer")

    if num_bins <= 0:
        raise ValueError("Number of bins must be positive")

    confidences = _validate_numeric_array(confidences, "Confidences")
    predictions = _validate_numeric_array(predictions, "Predictions")
    actual_labels = _validate_numeric_array(actual_labels, "Actual labels")

    if not (
        len(confidences) == len(predictions) == len(actual_labels)
    ):
        raise ValueError("All input arrays must have equal lengths")

    if len(confidences) == 0:
        raise ValueError("Input arrays cannot be empty")

    confidences = confidences.astype(float)
    predictions = predictions.astype(float)
    actual_labels = actual_labels.astype(float)

    if np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("Confidences must be between 0 and 1")

    if not np.all(predictions == np.floor(predictions)):
        raise ValueError("Predictions must be integer class labels")

    if not np.all(actual_labels == np.floor(actual_labels)):
        raise ValueError("Actual labels must be integer class labels")

    int64_min = -(2 ** 63)
    int64_max_exclusive = 2 ** 63

    if (
        np.any(predictions < int64_min)
        or np.any(predictions >= int64_max_exclusive)
        or np.any(actual_labels < int64_min)
        or np.any(actual_labels >= int64_max_exclusive)
    ):
        raise ValueError("Class labels are outside the supported integer range")

    predictions = predictions.astype(np.int64)
    actual_labels = actual_labels.astype(np.int64)

    correct = (predictions == actual_labels).astype(float)

    bin_edges = np.linspace(0, 1, num_bins + 1)

    bin_results = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):

        if i == num_bins - 1:
            mask = (
                (confidences >= bin_edges[i])
                & (confidences <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidences >= bin_edges[i])
                & (confidences < bin_edges[i + 1])
            )

        count = int(np.sum(mask))

        if count == 0:
            continue

        average_confidence = float(np.mean(confidences[mask]))
        accuracy = float(np.mean(correct[mask]))

        calibration_error = abs(accuracy - average_confidence)

        ece += (count / len(confidences)) * calibration_error
        mce = max(mce, calibration_error)

        bin_results.append({
            "bin": i + 1,
            "lower_bound": float(bin_edges[i]),
            "upper_bound": float(bin_edges[i + 1]),
            "count": count,
            "average_confidence": average_confidence,
            "accuracy": accuracy,
            "calibration_error": float(calibration_error)
        })

    brier_score = float(np.mean((confidences - correct) ** 2))

    average_confidence = float(np.mean(confidences))
    accuracy = float(np.mean(correct))

    if average_confidence > accuracy:
        reliability = "Overconfident"
    elif average_confidence < accuracy:
        reliability = "Underconfident"
    else:
        reliability = "Well Balanced"

    return {
        "total_samples": len(confidences),
        "accuracy": accuracy,
        "average_confidence": average_confidence,
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "brier_score": brier_score,
        "reliability": reliability,
        "calibration_bins": bin_results
    }