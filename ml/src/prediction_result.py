def format_prediction(
    class_id,
    confidence,
    class_names,
    confidence_threshold=0.60
):

    if not 0 <= class_id < len(class_names):
        raise ValueError("Invalid class ID")

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")

    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Invalid confidence threshold")

    class_name = class_names[class_id]

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