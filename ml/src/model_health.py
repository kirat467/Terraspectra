def assess_model_health(
    performance_metrics,
    average_confidence=None,
    drift_detected=False
):
    if not isinstance(performance_metrics, dict):
        raise ValueError("Performance metrics must be a dictionary")

    required = ["accuracy", "precision", "recall", "f1_score"]

    for metric in required:
        if metric not in performance_metrics:
            raise ValueError(f"Missing metric: {metric}")

    metrics = {}

    for metric in required:
        value = performance_metrics[metric]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid {metric}")

        metrics[metric] = float(value)

    if average_confidence is not None:
        if (
            isinstance(average_confidence, bool)
            or not isinstance(average_confidence, (int, float))
            or not 0 <= average_confidence <= 1
        ):
            raise ValueError("Invalid average confidence")

    if not isinstance(drift_detected, bool):
        raise ValueError("Drift status must be boolean")

    score = sum(metrics.values()) / len(metrics)

    recommendations = []

    if metrics["accuracy"] < 0.7:
        recommendations.append("Improve model accuracy.")

    if metrics["precision"] < 0.7:
        recommendations.append("Review false positive predictions.")

    if metrics["recall"] < 0.7:
        recommendations.append("Review false negative predictions.")

    if metrics["f1_score"] < 0.7:
        recommendations.append("Improve the balance of precision and recall.")

    if average_confidence is not None and average_confidence < 0.6:
        recommendations.append("Review low-confidence predictions.")

    if drift_detected:
        recommendations.append("Investigate model drift.")

    if not recommendations:
        recommendations.append("Continue monitoring model performance.")

    if drift_detected or score < 0.5:
        status = "Needs Attention"
    elif score < 0.7:
        status = "Moderate"
    else:
        status = "Good"

    return {
        "health_status": status,
        "health_score": round(score * 100, 2),
        "performance_metrics": metrics,
        "average_confidence": average_confidence,
        "drift_detected": drift_detected,
        "recommendations": recommendations
    }