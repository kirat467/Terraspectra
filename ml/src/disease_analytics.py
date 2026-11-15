import math
from collections import defaultdict


def generate_disease_analytics(predictions):

    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list")

    disease_data = defaultdict(list)

    for item in predictions:

        if not isinstance(item, dict):
            raise ValueError("Invalid prediction record")

        disease = item.get("prediction")
        confidence = item.get("confidence")

        if not isinstance(disease, str) or not disease.strip():
            raise ValueError("Invalid disease name")

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 100
        ):
            raise ValueError("Invalid confidence value")

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