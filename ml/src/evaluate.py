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

    if model is None:
        raise ValueError("Model cannot be None")

    if dataloader is None:
        raise ValueError("Dataloader cannot be None")

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = model.to(device)
    model.eval()

    actual = []
    predicted = []
    confidences = []

    top3_correct = 0
    total_samples = 0
    num_classes = None

    with torch.no_grad():

        for images, labels in dataloader:

            if not isinstance(images, torch.Tensor):
                raise ValueError("Evaluation images must be tensors")

            if not isinstance(labels, torch.Tensor):
                raise ValueError("Evaluation labels must be tensors")

            if images.ndim < 2:
                raise ValueError("Evaluation images have invalid dimensions")

            if images.shape[0] == 0:
                raise ValueError("Evaluation batch cannot be empty")

            # Validate image values, including NaN and infinity.
            if not torch.isfinite(images).all().item():
                raise ValueError(
                    "Evaluation images must contain finite values"
                )

            if labels.ndim != 1:
                raise ValueError("Evaluation labels must be 1D")

            if labels.shape[0] != images.shape[0]:
                raise ValueError(
                    "Image and label batch sizes must match"
                )

            if labels.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8
            ):
                raise ValueError(
                    "Evaluation labels must contain integers"
                )

            images = images.to(device)
            labels_device = labels.to(device).long()

            outputs = model(images)

            if not isinstance(outputs, torch.Tensor):
                raise ValueError("Model outputs must be tensors")

            if outputs.ndim != 2:
                raise ValueError("Model output must be a 2D tensor")

            if outputs.shape[0] != images.shape[0]:
                raise ValueError(
                    "Model output batch size must match input batch size"
                )

            if outputs.shape[1] == 0:
                raise ValueError(
                    "Model must return at least one class"
                )

            if not torch.isfinite(outputs).all().item():
                raise ValueError(
                    "Model outputs must contain finite values"
                )

            if num_classes is None:
                num_classes = outputs.shape[1]

            elif num_classes != outputs.shape[1]:
                raise ValueError(
                    "Inconsistent number of model classes"
                )

            if torch.any(labels_device < 0) or torch.any(
                labels_device >= num_classes
            ):
                raise ValueError(
                    "Actual labels outside model class range"
                )

            probabilities = torch.softmax(outputs, dim=1)

            if not torch.isfinite(probabilities).all().item():
                raise ValueError(
                    "Model probabilities must be finite"
                )

            confidence, predictions = probabilities.max(dim=1)

            top_k = min(3, num_classes)

            top3_predictions = probabilities.topk(
                top_k,
                dim=1
            ).indices

            top3_correct += (
                top3_predictions == labels_device.unsqueeze(1)
            ).any(dim=1).sum().item()

            total_samples += labels.size(0)

            actual.extend(
                labels_device.cpu().numpy().tolist()
            )

            predicted.extend(
                predictions.cpu().numpy().tolist()
            )

            confidences.extend(
                confidence.cpu().numpy().tolist()
            )

    if not actual:
        raise ValueError("Evaluation dataset is empty")

    class_ids = list(range(num_classes))

    if num_classes == len(CLASS_LABELS):

        class_names = [
            name for name, label in sorted(
                CLASS_LABELS.items(),
                key=lambda item: item[1]
            )
        ]

    else:
        class_names = [
            str(i) for i in class_ids
        ]

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

    top3_accuracy = top3_correct / total_samples

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

    confused_pairs = []

    for actual_id in class_ids:

        for predicted_id in class_ids:

            if (
                actual_id != predicted_id
                and matrix[actual_id][predicted_id] > 0
            ):

                confused_pairs.append({
                    "actual_class": class_names[actual_id],
                    "predicted_class": class_names[predicted_id],
                    "count": int(
                        matrix[actual_id][predicted_id]
                    )
                })

    confused_pairs.sort(
        key=lambda item: item["count"],
        reverse=True
    )

    mean_confidence = sum(confidences) / len(confidences)

    print("\n--- TerraSpectra Model Evaluation ---")
    print("Accuracy:", accuracy)
    print("Top-3 Accuracy:", top3_accuracy)
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
        "top3_accuracy": top3_accuracy,
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