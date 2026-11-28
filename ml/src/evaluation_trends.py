import math
from numbers import Real


def analyze_evaluation_trends(history):

    if not isinstance(history, list):
        raise TypeError("History must be a list")

    if not history:
        raise ValueError("History cannot be empty")

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]

    validated_history = []

    for entry in history:

        if not isinstance(entry, dict):
            raise TypeError("Each history entry must be a dictionary")

        validated_entry = {}

        for metric in metrics:

            if metric not in entry:
                raise ValueError(f"Missing metric: {metric}")

            value = entry[metric]

            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{metric} must be numeric")

            try:
                value = float(value)
            except (ValueError, OverflowError):
                raise ValueError(f"Invalid value for {metric}")

            if not math.isfinite(value):
                raise ValueError(f"{metric} must be finite")

            if not 0 <= value <= 1:
                raise ValueError(f"{metric} must be between 0 and 1")

            validated_entry[metric] = value

        validated_history.append(validated_entry)

    trends = {}

    for metric in metrics:

        values = [
            entry[metric]
            for entry in validated_history
        ]

        first_value = values[0]
        latest_value = values[-1]

        if len(values) == 1:
            trend = "Insufficient data"

        elif latest_value > first_value:
            trend = "Improving"

        elif latest_value < first_value:
            trend = "Declining"

        else:
            trend = "Stable"

        trends[metric] = {
            "values": values,
            "first_value": first_value,
            "latest_value": latest_value,
            "change": round(latest_value - first_value, 4),
            "trend": trend
        }

    return trends