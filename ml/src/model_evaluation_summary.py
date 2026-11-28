import math
from numbers import Real, Integral


def create_evaluation_summary(metrics):

    if not isinstance(metrics, dict):
        raise TypeError("Metrics must be a dictionary")

    required = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "test_samples"
    ]

    summary = {}

    for key in required:

        if key not in metrics:
            raise ValueError(f"Missing metric: {key}")

        value = metrics[key]

        if key == "test_samples":

            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("test_samples must be an integer")

            if value < 0:
                raise ValueError("test_samples must be non-negative")

            summary[key] = int(value)
            continue

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

        summary[key] = round(value, 4)

    return summary