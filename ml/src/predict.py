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
        raise ValueError("Confidence threshold must be between 0 and 100")

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
        confidence, prediction = torch.max(probabilities, dim=1)

    results = []

    for i in range(len(prediction)):

        class_id = prediction[i].item()
        confidence_percent = round(confidence[i].item() * 100, 2)

        is_confident = confidence_percent >= confidence_threshold

        results.append({
            "class_id": class_id,
            "prediction": class_names[class_id],
            "predicted_class": class_names[class_id],
            "confidence": confidence_percent,
            "status": "Confident" if is_confident else "Uncertain",
            "review_required": not is_confident
        })

    return results