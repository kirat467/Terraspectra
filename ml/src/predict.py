
import torch
import torch.nn.functional as F


def predict_crop(model, image, class_names):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)

        confidence, prediction = torch.max(
            probabilities, dim=1
        )

    results = []

    for i in range(len(prediction)):
        results.append({
            "prediction": class_names[prediction[i].item()],
            "confidence": round(
                confidence[i].item() * 100, 2
            )
        })

    return results
