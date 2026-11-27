from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = {
    0: "Healthy",
    1: "Alternaria alternata",
    2: "Alternaria solani",
    3: "Botrytis cinerea",
    4: "Fusarium oxysporum"
}


def visualize_predictions(results, output_dir="ml/results"):

    if not isinstance(results, dict):
        raise ValueError("Results must be a dictionary")

    required = [
        "confusion_matrix",
        "class_labels",
        "prediction_counts",
        "mean_confidence"
    ]

    for key in required:
        if key not in results:
            raise ValueError(f"Missing evaluation result: {key}")

    labels = results["class_labels"]

    if not isinstance(labels, (list, tuple)) or not labels:
        raise ValueError("Class labels must be a non-empty list")

    if any(
        isinstance(label, bool) or not isinstance(label, (int, np.integer))
        for label in labels
    ):
        raise ValueError("Class labels must contain integers")

    if len(set(labels)) != len(labels):
        raise ValueError("Class labels must be unique")

    try:
        matrix = np.asarray(results["confusion_matrix"])
    except (TypeError, ValueError):
        raise ValueError("Invalid confusion matrix")

    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Invalid confusion matrix shape")

    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("Confusion matrix must contain numbers")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Confusion matrix must contain finite values")

    if np.any(matrix < 0):
        raise ValueError("Confusion matrix cannot contain negative values")

    if not np.all(matrix == np.floor(matrix)):
        raise ValueError("Confusion matrix must contain whole numbers")

    counts = results["prediction_counts"]

    if not isinstance(counts, dict):
        raise ValueError("Prediction counts must be a dictionary")

    if set(counts.keys()) != set(labels):
        raise ValueError("Prediction counts must match class labels")

    for label, count in counts.items():
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 0
        ):
            raise ValueError("Prediction counts must be non-negative integers")

    predicted_counts = matrix.sum(axis=0)

    if not np.array_equal(
        predicted_counts,
        np.array([counts[label] for label in labels])
    ):
        raise ValueError(
            "Prediction counts do not match the confusion matrix"
        )

    confidence = results["mean_confidence"]

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float, np.integer, np.floating))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Confidence must be a finite number between 0 and 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [
        CLASS_NAMES.get(int(label), str(label))
        for label in labels
    ]

    actual_counts = matrix.sum(axis=1)

    x = np.arange(len(labels))
    width = 0.35

    # Actual vs predicted class comparison
    comparison_path = output_dir / "class_comparison.png"

    plt.figure(figsize=(12, 6))

    try:
        plt.bar(x - width / 2, actual_counts, width, label="Actual")
        plt.bar(x + width / 2, predicted_counts, width, label="Predicted")
        plt.xticks(x, names, rotation=25, ha="right")
        plt.ylabel("Number of patches")
        plt.title("Actual vs Predicted Classes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(comparison_path)
    finally:
        plt.close()

    # Confusion matrix visualization
    matrix_path = output_dir / "confusion_matrix.png"

    plt.figure(figsize=(8, 6))

    try:
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar(label="Number of patches")
        plt.xticks(range(len(names)), names, rotation=25, ha="right")
        plt.yticks(range(len(names)), names)
        plt.xlabel("Predicted class")
        plt.ylabel("Actual class")
        plt.title("TerraSpectra Confusion Matrix")

        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                plt.text(
                    col,
                    row,
                    str(int(matrix[row, col])),
                    ha="center",
                    va="center"
                )

        plt.tight_layout()
        plt.savefig(matrix_path)
    finally:
        plt.close()

    # Average prediction confidence
    confidence_path = output_dir / "prediction_confidence.png"

    plt.figure(figsize=(7, 4))

    try:
        plt.bar(["Average Confidence"], [confidence])
        plt.ylim(0, 1)
        plt.ylabel("Confidence")
        plt.title("Model Prediction Confidence")
        plt.tight_layout()
        plt.savefig(confidence_path)
    finally:
        plt.close()

    print("Prediction visualizations saved successfully!")

    return {
        "class_comparison": str(comparison_path),
        "confusion_matrix": str(matrix_path),
        "prediction_confidence": str(confidence_path)
    }