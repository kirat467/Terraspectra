
import torch
import torch.nn.functional as F


def predict_crop(model, image, class_names, device=None):

    if not class_names:
        raise ValueError("Class names cannot be empty")

    if image.ndim < 2:
        raise ValueError("Invalid image tensor")

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
        results.append({
            "class_id": prediction[i].item(),
            "prediction": class_names[prediction[i].item()],
            "confidence": round(confidence[i].item() * 100, 2)
        })

    return results
