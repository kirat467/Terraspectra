def analyze_model_reliability(predictions, confidence_threshold=0.60):
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Predictions must be a non-empty list")

    if not isinstance(confidence_threshold, (int, float)) or isinstance(confidence_threshold, bool) or not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1")

    for confidence in predictions:
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise ValueError("Every confidence value must be between 0 and 1")

    total = len(predictions)
    reliable = sum(c >= confidence_threshold for c in predictions)
    uncertain = total - reliable

    average = sum(predictions) / total

    recommendations = []

    if average < 0.60:
        recommendations.append("Improve model prediction confidence.")

    if uncertain > reliable:
        recommendations.append("Review uncertain predictions manually.")

    if not recommendations:
        recommendations.append("Model reliability is satisfactory.")

    return {
        "total_predictions": total,
        "reliable_predictions": reliable,
        "uncertain_predictions": uncertain,
        "reliability_percentage": round(reliable / total * 100, 2),
        "average_confidence": round(average, 4),
        "confidence_threshold": confidence_threshold,
        "recommendations": recommendations
    }