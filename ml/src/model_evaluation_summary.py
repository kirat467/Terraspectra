def create_evaluation_summary(metrics):
    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary")

    required = ["accuracy", "precision", "recall", "f1_score", "test_samples"]

    for key in required:
        if key not in metrics:
            raise ValueError(f"Missing metric: {key}")

    summary = {}

    for key in required:
        value = metrics[key]

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{key} must be numeric")

        if key != "test_samples" and not 0 <= value <= 1:
            raise ValueError(f"{key} must be between 0 and 1")

        if key == "test_samples" and (value < 0 or not isinstance(value, int)):
            raise ValueError("test_samples must be a non-negative integer")

        summary[key] = round(value, 4) if key != "test_samples" else value

    return summary