def analyze_prediction_errors(y_true, y_pred):
    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError("Inputs must be lists or tuples")

    if not y_true or not y_pred:
        raise ValueError("Inputs cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("Both inputs must have equal length")

    classes = sorted(set(y_true) | set(y_pred))

    correct = sum(a == b for a, b in zip(y_true, y_pred))
    incorrect = len(y_true) - correct

    class_analysis = {}

    for cls in classes:
        total = sum(actual == cls for actual in y_true)
        errors = sum(
            actual == cls and predicted != cls
            for actual, predicted in zip(y_true, y_pred)
        )

        class_analysis[cls] = {
            "total_samples": total,
            "correct_predictions": total - errors,
            "incorrect_predictions": errors,
            "error_rate": round(errors / total * 100, 2) if total else 0.0
        }

    misclassified = [
        {"actual": actual, "predicted": predicted}
        for actual, predicted in zip(y_true, y_pred)
        if actual != predicted
    ]

    return {
        "total_predictions": len(y_true),
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "accuracy": round(correct / len(y_true) * 100, 2),
        "error_rate": round(incorrect / len(y_true) * 100, 2),
        "class_analysis": class_analysis,
        "misclassified_samples": misclassified
    }