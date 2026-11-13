import numpy as np


def analyze_class_distribution(labels, num_classes):
    labels = np.asarray(labels)

    if labels.size == 0:
        raise ValueError("Labels cannot be empty")

    if num_classes <= 0:
        raise ValueError("Number of classes must be positive")

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