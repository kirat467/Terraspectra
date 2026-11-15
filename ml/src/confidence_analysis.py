def classify_confidence(confidence):

    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 100
    ):
        raise ValueError("Confidence must be between 0 and 100")

    if confidence < 40:
        level = "Low"
    elif confidence < 70:
        level = "Medium"
    else:
        level = "High"

    return {
        "confidence": round(confidence, 2),
        "confidence_level": level
    }


def analyze_prediction_confidence(predictions):

    if not isinstance(predictions, list):
        raise ValueError("Predictions must be a list")

    results = []

    for prediction in predictions:

        if not isinstance(prediction, dict):
            raise ValueError("Invalid prediction record")

        if "confidence" not in prediction:
            raise ValueError("Confidence value is missing")

        result = prediction.copy()
        result.update(
            classify_confidence(prediction["confidence"])
        )

        results.append(result)

    return results