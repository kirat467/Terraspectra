def analyze_evaluation_trends(history):
    if not isinstance(history, list):
        raise TypeError("History must be a list")

    if not history:
        raise ValueError("History cannot be empty")

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    for entry in history:
        if not isinstance(entry, dict):
            raise TypeError("Each history entry must be a dictionary")

        for metric in metrics:
            if metric not in entry:
                raise ValueError(f"Missing metric: {metric}")

            value = entry[metric]

            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{metric} must be numeric")

            if not 0 <= value <= 1:
                raise ValueError(f"{metric} must be between 0 and 1")

    trends = {}

    for metric in metrics:
        values = [entry[metric] for entry in history]

        if len(values) == 1:
            trend = "Insufficient data"
        elif values[-1] > values[0]:
            trend = "Improving"
        elif values[-1] < values[0]:
            trend = "Declining"
        else:
            trend = "Stable"

        trends[metric] = {
            "values": values,
            "first_value": values[0],
            "latest_value": values[-1],
            "change": round(values[-1] - values[0], 4),
            "trend": trend
        }

    return trends