import numpy as np


def analyze_class_distribution(labels, num_classes):
    """Analyze class distribution and calculate class weights."""

    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
        num_classes, (int, np.integer)
    ):
        raise ValueError("Number of classes must be a positive integer")

    if num_classes <= 0:
        raise ValueError("Number of classes must be positive")

    try:
        labels = np.asarray(labels)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be a valid one-dimensional array") from exc

    if labels.size == 0:
        raise ValueError("Labels cannot be empty")

    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional")

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Labels must be integers")

    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Invalid class labels")

    counts = np.bincount(labels, minlength=num_classes)

    total = len(labels)
    weights = np.zeros(num_classes, dtype=np.float32)

    present = counts > 0
    weights[present] = total / (num_classes * counts[present])

    distribution = {
        class_id: int(counts[class_id])
        for class_id in range(num_classes)
    }

    print("Class Distribution:", distribution)
    print("Class Weights:", weights.tolist())

    return {
        "class_counts": distribution,
        "class_weights": weights
    }