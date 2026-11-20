def analyze_class_performance(y_true, y_pred):
    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError("Inputs must be lists or tuples")

    if not y_true or not y_pred:
        raise ValueError("Inputs cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("Both inputs must have equal length")

    classes = sorted(set(y_true) | set(y_pred), key=str)
    results = {}

    for cls in classes:
        tp = sum(a == cls and p == cls for a, p in zip(y_true, y_pred))
        fp = sum(a != cls and p == cls for a, p in zip(y_true, y_pred))
        fn = sum(a == cls and p != cls for a, p in zip(y_true, y_pred))

        support = sum(a == cls for a in y_true)

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        results[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support
        }

    best_class = max(results, key=lambda c: results[c]["f1_score"])
    weakest_class = min(results, key=lambda c: results[c]["f1_score"])

    return {
        "class_performance": results,
        "best_performing_class": best_class,
        "weakest_performing_class": weakest_class
    }