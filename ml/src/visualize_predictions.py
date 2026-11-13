from pathlib import Path
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
    required = [
        "confusion_matrix",
        "class_labels",
        "prediction_counts",
        "mean_confidence"
    ]

    for key in required:
        if key not in results:
            raise ValueError(f"Missing evaluation result: {key}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = results["class_labels"]
    matrix = np.asarray(results["confusion_matrix"])
    counts = results["prediction_counts"]

    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Invalid confusion matrix shape")

    names = [CLASS_NAMES.get(int(label), str(label)) for label in labels]

    actual_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)

    # Actual vs predicted class comparison
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, actual_counts, width, label="Actual")
    plt.bar(x + width / 2, predicted_counts, width, label="Predicted")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("Number of patches")
    plt.title("Actual vs Predicted Classes")
    plt.legend()
    plt.tight_layout()

    comparison_path = output_dir / "class_comparison.png"
    plt.savefig(comparison_path)
    plt.close()

    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Number of patches")
    plt.xticks(range(len(names)), names, rotation=25, ha="right")
    plt.yticks(range(len(names)), names)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("TerraSpectra Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, str(matrix[row, col]),
                     ha="center", va="center")

    plt.tight_layout()

    matrix_path = output_dir / "confusion_matrix.png"
    plt.savefig(matrix_path)
    plt.close()

    # Average prediction confidence
    confidence = results["mean_confidence"]

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")

    plt.figure(figsize=(7, 4))
    plt.bar(["Average Confidence"], [confidence])
    plt.ylim(0, 1)
    plt.ylabel("Confidence")
    plt.title("Model Prediction Confidence")
    plt.tight_layout()

    confidence_path = output_dir / "prediction_confidence.png"
    plt.savefig(confidence_path)
    plt.close()

    print("Prediction visualizations saved successfully!")

    return {
        "class_comparison": str(comparison_path),
        "confusion_matrix": str(matrix_path),
        "prediction_confidence": str(confidence_path)
    }