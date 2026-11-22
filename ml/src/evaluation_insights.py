def generate_evaluation_insights(metrics):
    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary")

    required = ["accuracy", "precision", "recall", "f1_score"]

    for key in required:
        if key not in metrics:
            raise ValueError(f"Missing metric: {key}")

        value = metrics[key]

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{key} must be numeric")

        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")

    insights = []

    if metrics["accuracy"] < 0.5:
        insights.append("Overall accuracy needs improvement")
    elif metrics["accuracy"] >= 0.8:
        insights.append("The model has strong overall accuracy")
    else:
        insights.append("Overall accuracy is moderate")

    if metrics["precision"] < metrics["recall"]:
        insights.append("The model produces relatively more false positives")
    elif metrics["precision"] > metrics["recall"]:
        insights.append("The model misses relatively more actual positive cases")
    else:
        insights.append("Precision and recall are balanced")

    if metrics["f1_score"] < 0.5:
        insights.append("F1-score indicates weak precision-recall balance")
    elif metrics["f1_score"] >= 0.8:
        insights.append("F1-score indicates strong precision-recall balance")
    else:
        insights.append("F1-score has room for improvement")

    return {
        "insights": insights,
        "strongest_metric": max(required, key=lambda key: metrics[key]),
        "weakest_metric": min(required, key=lambda key: metrics[key])
    }