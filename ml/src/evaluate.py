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
        actual, predicted,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual, predicted,
        labels=class_ids,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual, predicted,
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

    # Per-class performance
    per_class_metrics = {}

    for class_id, class_name in enumerate(class_names):

        class_data = report[class_name]

        correct = int(matrix[class_id][class_id])
        total = int(matrix[class_id].sum())
        incorrect = total - correct

        per_class_metrics[class_name] = {
            "accuracy": correct / total if total else 0.0,
            "precision": class_data["precision"],
            "recall": class_data["recall"],
            "f1_score": class_data["f1-score"],
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "total_samples": total
        }

    # Find most confused disease pairs
    confused_pairs = []

    for actual_id in class_ids:
        for predicted_id in class_ids:

            if actual_id != predicted_id and matrix[actual_id][predicted_id] > 0:

                confused_pairs.append({
                    "actual_class": class_names[actual_id],
                    "predicted_class": class_names[predicted_id],
                    "count": int(matrix[actual_id][predicted_id])
                })

    confused_pairs.sort(
        key=lambda item: item["count"],
        reverse=True
    )

    mean_confidence = sum(confidences) / len(confidences)

    print("\n--- TerraSpectra Model Evaluation ---")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\n--- Per-Class Performance ---")

    for name, metrics in per_class_metrics.items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")

    print("\nMost Confused Disease Pairs:")
    print(confused_pairs)

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
        "mean_confidence": float(mean_confidence),
        "per_class_metrics": per_class_metrics,
        "confused_pairs": confused_pairs
    }