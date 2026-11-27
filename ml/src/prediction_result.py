import math


def format_prediction(
    class_id,
    confidence,
    class_names,
    confidence_threshold=0.60
):
    if not isinstance(class_names, (list, tuple)):
        raise ValueError("Class names must be a list or tuple")

    if not class_names:
        raise ValueError("Class names cannot be empty")

    if isinstance(class_id, bool) or not isinstance(class_id, int):
        raise ValueError("Class ID must be an integer")

    if not 0 <= class_id < len(class_names):
        raise ValueError("Invalid class ID")

    if any(
        not isinstance(name, str) or not name.strip()
        for name in class_names
    ):
        raise ValueError("Class names must be non-empty strings")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError(
            "Confidence must be a finite number between 0 and 1"
        )

    if (
        isinstance(confidence_threshold, bool)
        or not isinstance(confidence_threshold, (int, float))
        or not math.isfinite(confidence_threshold)
        or not 0 <= confidence_threshold <= 1
    ):
        raise ValueError(
            "Confidence threshold must be a finite number between 0 and 1"
        )

    class_name = class_names[class_id].strip()

    is_confident = confidence >= confidence_threshold

    return {
        "class_id": class_id,
        "predicted_class": class_name,
        "confidence": round(confidence * 100, 2),
        "status": (
            "Confident"
            if is_confident
            else "Uncertain - Further analysis required"
        )
    }