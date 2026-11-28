import math
from numbers import Real
from collections import defaultdict


def generate_disease_analytics(predictions):

    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list")

    disease_data = defaultdict(list)

    for item in predictions:

        if not isinstance(item, dict):
            raise ValueError("Invalid prediction record")

        if "prediction" not in item or "confidence" not in item:
            raise ValueError("Prediction or confidence value is missing")

        disease = item["prediction"]
        confidence = item["confidence"]

        if not isinstance(disease, str) or not disease.strip():
            raise ValueError("Invalid disease name")

        if isinstance(confidence, bool) or not isinstance(confidence, Real):
            raise ValueError("Confidence must be numeric")

        try:
            confidence = float(confidence)
        except (ValueError, OverflowError):
            raise ValueError("Invalid confidence value")

        if not math.isfinite(confidence):
            raise ValueError("Confidence must be finite")

        if not 0 <= confidence <= 100:
            raise ValueError("Confidence must be between 0 and 100")

        disease = disease.strip()
        disease_data[disease].append(confidence)

    total = len(predictions)

    if total == 0:
        return {
            "total_predictions": 0,
            "most_detected_disease": None,
            "disease_analytics": []
        }

    analytics = []

    for disease, scores in disease_data.items():

        count = len(scores)

        analytics.append({
            "disease": disease,
            "count": count,
            "percentage": round(count / total * 100, 2),
            "average_confidence": round(sum(scores) / count, 2),
            "highest_confidence": max(scores),
            "lowest_confidence": min(scores)
        })

    analytics.sort(
        key=lambda item: item["count"],
        reverse=True
    )

    return {
        "total_predictions": total,
        "most_detected_disease": analytics[0]["disease"],
        "disease_analytics": analytics
    }