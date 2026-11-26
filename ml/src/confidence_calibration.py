import numpy as np


def calculate_confidence_calibration(
    confidences,
    predictions,
    actual_labels,
    num_bins=10
):

    try:
        confidences = np.asarray(confidences, dtype=float)
        predictions = np.asarray(predictions)
        actual_labels = np.asarray(actual_labels)

    except (TypeError, ValueError) as error:
        raise ValueError("Inputs must contain valid numeric values") from error

    if confidences.ndim != 1:
        raise ValueError("Confidences must be 1D")

    if predictions.ndim != 1:
        raise ValueError("Predictions must be 1D")

    if actual_labels.ndim != 1:
        raise ValueError("Actual labels must be 1D")

    if not (
        len(confidences) == len(predictions) == len(actual_labels)
    ):
        raise ValueError("All input arrays must have equal lengths")

    if len(confidences) == 0:
        raise ValueError("Input arrays cannot be empty")

    if not isinstance(num_bins, (int, np.integer)) or isinstance(
        num_bins, (bool, np.bool_)
    ):
        raise ValueError("Number of bins must be a positive integer")

    if num_bins <= 0:
        raise ValueError("Number of bins must be positive")

    if not np.all(np.isfinite(confidences)):
        raise ValueError("Confidences must contain finite values")

    if np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("Confidences must be between 0 and 1")

    try:
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Predictions must contain finite values")

        if not np.all(np.isfinite(actual_labels)):
            raise ValueError("Actual labels must contain finite values")

        if not np.all(predictions == predictions.astype(int)):
            raise ValueError("Predictions must be integer class labels")

        if not np.all(actual_labels == actual_labels.astype(int)):
            raise ValueError("Actual labels must be integer class labels")

    except (TypeError, ValueError, OverflowError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            ("Predictions", "Actual labels")
        ):
            raise
        raise ValueError("Labels must contain valid integer values") from error

    predictions = predictions.astype(int)
    actual_labels = actual_labels.astype(int)

    correct = (predictions == actual_labels).astype(float)

    bin_edges = np.linspace(0, 1, num_bins + 1)

    bin_results = []
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):

        if i == num_bins - 1:
            mask = (
                (confidences >= bin_edges[i]) &
                (confidences <= bin_edges[i + 1])
            )

        else:
            mask = (
                (confidences >= bin_edges[i]) &
                (confidences < bin_edges[i + 1])
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