import torch
import torch.nn.functional as F


def predict_crop(
    model,
    image,
    class_names,
    device=None,
    confidence_threshold=60.0
):
    if not class_names:
        raise ValueError("Class names cannot be empty")

    if image.ndim < 2:
        raise ValueError("Invalid image tensor")

    if not 0 <= confidence_threshold <= 100:
        raise ValueError("Invalid confidence threshold")

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()
    image = image.to(device)

    with torch.inference_mode():
        outputs = model(image)

        if outputs.ndim != 2:
            raise ValueError("Model must return 2D class logits")

        if outputs.shape[1] != len(class_names):
            raise ValueError("Class names do not match model outputs")

        probabilities = F.softmax(outputs, dim=1)

        top_count = min(3, len(class_names))
        top_probs, top_indices = torch.topk(
            probabilities, top_count, dim=1
        )

    results = []

    for i in range(len(top_indices)):
        top_predictions = []

        for j in range(top_count):
            class_id = top_indices[i][j].item()
            confidence = round(top_probs[i][j].item() * 100, 2)

            top_predictions.append({
                "class_id": class_id,
                "disease": class_names[class_id],
                "confidence": confidence
            })

        best = top_predictions[0]

        results.append({
            "class_id": best["class_id"],
            "prediction": best["disease"],
            "predicted_class": best["disease"],
            "confidence": best["confidence"],
            "status": (
                "Confident"
                if best["confidence"] >= confidence_threshold
                else "Uncertain"
            ),
            "review_required": best["confidence"] < confidence_threshold,
            "top_predictions": top_predictions
        })

    return results