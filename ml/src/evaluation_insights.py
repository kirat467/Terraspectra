import math
from numbers import Real


def generate_evaluation_insights(metrics):

    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary")

    required = [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    validated_metrics = {}

    for key in required:

        if key not in metrics:
            raise ValueError(f"Missing metric: {key}")

        value = metrics[key]

        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{key} must be numeric")

        try:
            value = float(value)
        except (ValueError, OverflowError):
            raise ValueError(f"Invalid value for {key}")

        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")

        if not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")

        validated_metrics[key] = value

    insights = []

    accuracy = validated_metrics["accuracy"]
    precision = validated_metrics["precision"]
    recall = validated_metrics["recall"]
    f1_score = validated_metrics["f1_score"]

    if accuracy < 0.5:
        insights.append("Overall accuracy needs improvement")

    elif accuracy >= 0.8:
        insights.append("The model has strong overall accuracy")

    else:
        insights.append("Overall accuracy is moderate")

    if precision < recall:
        insights.append(
            "The model produces relatively more false positives"
        )

    elif precision > recall:
        insights.append(
            "The model misses relatively more actual positive cases"
        )

    else:
        insights.append("Precision and recall are balanced")

    if f1_score < 0.5:
        insights.append(
            "F1-score indicates weak precision-recall balance"
        )

    elif f1_score >= 0.8:
        insights.append(
            "F1-score indicates strong precision-recall balance"
        )

    else:
        insights.append("F1-score has room for improvement")

    return {
        "insights": insights,
        "strongest_metric": max(
            required,
            key=lambda key: validated_metrics[key]
        ),
        "weakest_metric": min(
            required,
            key=lambda key: validated_metrics[key]
        )
    }