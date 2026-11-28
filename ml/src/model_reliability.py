import math
from numbers import Real


def analyze_model_reliability(predictions, confidence_threshold=0.60):

    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Predictions must be a non-empty list")

    if (
        isinstance(confidence_threshold, bool)
        or not isinstance(confidence_threshold, Real)
    ):
        raise TypeError("Confidence threshold must be numeric")

    confidence_threshold = float(confidence_threshold)

    if not math.isfinite(confidence_threshold):
        raise ValueError("Confidence threshold must be finite")

    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1")

    validated_predictions = []

    for confidence in predictions:

        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise TypeError("Every confidence value must be numeric")

        confidence = float(confidence)

        if not math.isfinite(confidence):
            raise ValueError("Confidence values must be finite")

        if not 0 <= confidence <= 1:
            raise ValueError("Every confidence value must be between 0 and 1")

        validated_predictions.append(confidence)

    total = len(validated_predictions)

    reliable = sum(
        confidence >= confidence_threshold
        for confidence in validated_predictions
    )

    uncertain = total - reliable

    average = sum(validated_predictions) / total

    recommendations = []

    if average < 0.60:
        recommendations.append(
            "Improve model prediction confidence."
        )

    if uncertain > reliable:
        recommendations.append(
            "Review uncertain predictions manually."
        )

    if not recommendations:
        recommendations.append(
            "Model reliability is satisfactory."
        )

    return {
        "total_predictions": total,
        "reliable_predictions": reliable,
        "uncertain_predictions": uncertain,
        "reliability_percentage": round(reliable / total * 100, 2),
        "average_confidence": round(average, 4),
        "confidence_threshold": confidence_threshold,
        "recommendations": recommendations
    }