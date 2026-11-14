import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from label_mapping import CLASS_LABELS


def evaluate_model(model, dataloader, device=None):

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    actual = []
    predicted = []
    confidences = []

    num_classes = None

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)

            outputs = model(images)

            if num_classes is None:
                num_classes = outputs.shape[1]

            elif num_classes != outputs.shape[1]:
                raise ValueError("Inconsistent number of model classes")

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predictions = probabilities.max(dim=1)

            actual.extend(labels.cpu().numpy().tolist())
            predicted.extend(predictions.cpu().numpy().tolist())
            confidences.extend(confidence.cpu().numpy().tolist())

    if not actual:
        raise ValueError("Evaluation dataset is empty")

    class_ids = list(range(num_classes))

    if any(label not in class_ids for label in actual):
        raise ValueError("Actual labels outside model class range")

    if num_classes == len(CLASS_LABELS):
        class_names = [
            name for name, label in sorted(
                CLASS_LABELS.items(),
                key=lambda item: item[1]
            )
        ]
    else:
        class_names = [str(i) for i in class_ids]

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual,
        predicted,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        actual,
        predicted,
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=class_ids
    )

    prediction_counts = {
        class_id: predicted.count(class_id)
        for class_id in class_ids
    }

    mean_confidence = sum(confidences) / len(confidences)

    print("\n--- TerraSpectra Model Evaluation ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nPrediction Distribution:")
    print(prediction_counts)

    print("\nAverage Prediction Confidence:", mean_confidence)

    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=class_ids,
            target_names=class_names,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(matrix)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_labels": class_ids,
        "class_names": class_names,
        "prediction_counts": prediction_counts,
        "mean_confidence": float(mean_confidence)
    }