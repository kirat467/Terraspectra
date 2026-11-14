from collections import Counter


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
            "diseased_percentage": 0
        }

    disease_counts = Counter()
    healthy_count = 0
    confidence_values = []

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
            or not 0 <= confidence <= 100
        ):
            raise ValueError("Invalid confidence value")

        disease_counts[disease] += 1
        confidence_values.append(confidence)

        if disease.strip().lower() == "healthy":
            healthy_count += 1

    total = len(predictions)
    diseased_count = total - healthy_count

    most_common = disease_counts.most_common(1)[0][0]

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
        )
    }