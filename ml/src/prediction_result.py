
def format_prediction(class_id, confidence, class_names):
    if not 0 <= class_id < len(class_names):
        raise ValueError("Invalid class ID")

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")

    class_name = class_names[class_id]

    return {
        "class_id": class_id,
        "predicted_class": class_name,
        "confidence": round(confidence * 100, 2),
        "status": "Prediction generated"
    }
