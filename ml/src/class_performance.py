import math


def analyze_class_performance(y_true, y_pred):
    """Calculate precision, recall, F1-score, and support for each class."""

    if not isinstance(y_true, (list, tuple)) or not isinstance(
        y_pred, (list, tuple)
    ):
        raise ValueError("Inputs must be lists or tuples")

    if not y_true or not y_pred:
        raise ValueError("Inputs cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("Both inputs must have equal length")

    for label in list(y_true) + list(y_pred):
        try:
            hash(label)
        except TypeError as exc:
            raise ValueError("Labels must be hashable") from exc

        if label is None:
            raise ValueError("Labels cannot contain missing values")

        if isinstance(label, float) and math.isnan(label):
            raise ValueError("Labels cannot contain missing values")

    classes = sorted(set(y_true) | set(y_pred), key=str)
    results = {}

    for cls in classes:
        tp = sum(
            actual == cls and predicted == cls
            for actual, predicted in zip(y_true, y_pred)
        )

        fp = sum(
            actual != cls and predicted == cls
            for actual, predicted in zip(y_true, y_pred)
        )

        fn = sum(
            actual == cls and predicted != cls
            for actual, predicted in zip(y_true, y_pred)
        )

        support = sum(actual == cls for actual in y_true)

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        results[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

    best_class = max(
        results,
        key=lambda cls: results[cls]["f1_score"]
    )

    weakest_class = min(
        results,
        key=lambda cls: results[cls]["f1_score"]
    )

    return {
        "class_performance": results,
        "best_performing_class": best_class,
        "weakest_performing_class": weakest_class
    }