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

    # Validate input dimensions
    if image.ndim not in (3, 5):
        raise ValueError(
            "Input image must be a 5D tensor: "
            "[batch, channels, depth, height, width] "
            "or a 3D ViT tensor"
        )

    # Validate empty batch
    if image.shape[0] == 0:
        raise ValueError("Prediction batch cannot be empty")

    # Validate ViT input dimensions
    if image.ndim == 3:
        if image.shape[1] == 0 or image.shape[2] == 0:
            raise ValueError("Input image dimensions cannot be empty")

    # Validate CNN input dimensions
    if image.ndim == 5:
        if image.shape[1] != 1:
            raise ValueError("Input image must have exactly 1 channel")

        if any(size == 0 for size in image.shape[2:]):
            raise ValueError("Input image dimensions cannot be empty")

    # Validate finite input values
    if not torch.isfinite(image).all():
        raise ValueError("Input image must contain finite values")

    # Validate confidence threshold
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

        # Validate model output dimensions
        if outputs.ndim != 2:
            raise ValueError("Model must return 2D class logits")

        if outputs.shape[0] == 0:
            raise ValueError("Prediction batch cannot be empty")

        if outputs.shape[1] != len(class_names):
            raise ValueError("Class names do not match model outputs")

        if not torch.isfinite(outputs).all():
            raise ValueError("Model outputs must contain finite values")

        # Calculate probabilities
        probabilities = F.softmax(outputs, dim=1)

        if not torch.isfinite(probabilities).all():
            raise ValueError("Model probabilities must be finite")

        if not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones(probabilities.shape[0], device=device),
            atol=1e-5
        ):
            raise ValueError("Invalid prediction probabilities")

        # Get Top-3 predictions
        top_count = min(3, len(class_names))

        top_probs, top_indices = torch.topk(
            probabilities,
            top_count,
            dim=1
        )

    results = []

    for i in range(len(top_indices)):
        top_predictions = []

        for j in range(top_count):
            class_id = top_indices[i][j].item()
            confidence = round(
                top_probs[i][j].item() * 100,
                2
            )

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
            "review_required": (
                best["confidence"] < confidence_threshold
            ),
            "top_predictions": top_predictions
        })

    return results