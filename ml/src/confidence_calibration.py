import numpy as np


def calculate_confidence_calibration(
    confidences,
    predictions,
    actual_labels,
    num_bins=10
):
    confidences = np.asarray(confidences, dtype=float)
    predictions = np.asarray(predictions)
    actual_labels = np.asarray(actual_labels)

    if not (
        len(confidences) == len(predictions) == len(actual_labels)
    ):
        raise ValueError("All input arrays must have equal lengths")

    if len(confidences) == 0:
        raise ValueError("Input arrays cannot be empty")

    if not np.all(np.isfinite(confidences)):
        raise ValueError("Confidences must contain finite values")

    if np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("Confidences must be between 0 and 1")

    if num_bins <= 0:
        raise ValueError("Number of bins must be positive")

    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(actual_labels)):
        raise ValueError("Labels must contain finite values")

    if not np.all(predictions == predictions.astype(int)):
        raise ValueError("Predictions must be integer class labels")

    if not np.all(actual_labels == actual_labels.astype(int)):
        raise ValueError("Actual labels must be integer class labels")

    predictions = predictions.astype(int)
    actual_labels = actual_labels.astype(int)

    correct = (predictions == actual_labels).astype(float)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_results = []
    ece = 0.0

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

        ece += (count / len(confidences)) * abs(
            accuracy - average_confidence
        )

        bin_results.append({
            "bin": i + 1,
            "lower_bound": float(bin_edges[i]),
            "upper_bound": float(bin_edges[i + 1]),
            "count": count,
            "average_confidence": average_confidence,
            "accuracy": accuracy
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
        "brier_score": brier_score,
        "reliability": reliability,
        "calibration_bins": bin_results
    }