
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

            actual.extend(labels.numpy())
            predicted.extend(predictions.cpu().numpy())

    print("Accuracy:", accuracy_score(actual, predicted))
    print("Precision:", precision_score(
        actual, predicted, average="weighted", zero_division=0
    ))
    print("Recall:", recall_score(
        actual, predicted, average="weighted", zero_division=0
    ))
    print("F1 Score:", f1_score(
        actual, predicted, average="weighted", zero_division=0
    ))

    print("\nClassification Report:")
    print(classification_report(
        actual, predicted, zero_division=0
    ))
