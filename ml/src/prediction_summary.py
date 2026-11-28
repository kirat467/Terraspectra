from collections import Counter
from math import isfinite


def generate_prediction_summary(predictions):

    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list")

    if not predictions:
        return {
            "total_predictions": 0,
            "healthy_count": 0,
            "diseased_count": 0,
            "most_common_disease": None,
            "average_confidence": 0,
            "healthy_percentage": 0,
            "diseased_percentage": 0,
            "disease_distribution": {},
            "minimum_confidence": 0,
            "maximum_confidence": 0
        }

    disease_counts = Counter()
    confidence_values = []
    healthy_count = 0

    for item in predictions:

        if not isinstance(item, dict):
            raise ValueError("Invalid prediction record")

        disease = item.get("prediction")
        confidence = item.get("confidence")

        if not isinstance(disease, str) or not disease.strip():
            raise ValueError("Invalid disease name")

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(confidence)
            or not 0 <= confidence <= 100
        ):
            raise ValueError("Invalid confidence value")

        disease = disease.strip()

        disease_counts[disease] += 1
        confidence_values.append(confidence)

        if disease.lower() == "healthy":
            healthy_count += 1

    total = len(predictions)
    diseased_count = total - healthy_count

    most_common = disease_counts.most_common(1)[0][0]

    disease_distribution = {
        disease: {
            "count": count,
            "percentage": round(count / total * 100, 2)
        }
        for disease, count in disease_counts.items()
    }

    return {
        "total_predictions": total,
        "healthy_count": healthy_count,
        "diseased_count": diseased_count,
        "most_common_disease": most_common,
        "average_confidence": round(
            sum(confidence_values) / total, 2
        ),
        "healthy_percentage": round(
            healthy_count / total * 100, 2
        ),
        "diseased_percentage": round(
            diseased_count / total * 100, 2
        ),
        "disease_distribution": disease_distribution,
        "minimum_confidence": round(min(confidence_values), 2),
        "maximum_confidence": round(max(confidence_values), 2)
    }