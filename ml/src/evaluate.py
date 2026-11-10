
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluate_model(model, dataloader, device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for images, labels in dataloader:

            images = images.to(device)
            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            actual.extend(labels.detach().cpu().numpy())
            predicted.extend(predictions.detach().cpu().numpy())

    if not actual:
        raise ValueError("Evaluation dataset is empty")

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted, average="weighted", zero_division=0
    )

    recall = recall_score(
        actual, predicted, average="weighted", zero_division=0
    )

    f1 = f1_score(
        actual, predicted, average="weighted", zero_division=0
    )

    report = classification_report(
        actual, predicted, zero_division=0, output_dict=True
    )

    print("\n--- TerraSpectra Model Evaluation ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nClassification Report:")
    print(classification_report(
        actual, predicted, zero_division=0
    ))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report
    }
