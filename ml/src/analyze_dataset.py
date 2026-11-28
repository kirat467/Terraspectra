import numpy as np
from collections import Counter


def analyze_dataset(images, labels):
    """Analyze dataset dimensions, missing labels, and class distribution."""

    if images is None:
        raise ValueError("Images cannot be None")

    if labels is None:
        raise ValueError("Labels cannot be None")

    try:
        image_count = len(images)
        label_count = len(labels)
    except TypeError as exc:
        raise ValueError("Images and labels must be sized collections") from exc

    if image_count != label_count:
        raise ValueError("Images and labels must have equal length")

    if label_count == 0:
        raise ValueError("Dataset is empty")

    if isinstance(images, np.ndarray):
        if images.ndim < 2:
            raise ValueError("Images must have at least 2 dimensions")

        if not np.issubdtype(images.dtype, np.number):
            raise ValueError("Images must contain numeric values")

        if not np.isfinite(images).all():
            raise ValueError("Images must contain only finite values")

    missing = []

    for label in labels:
        is_missing = label is None

        if isinstance(label, (float, np.floating)):
            is_missing = is_missing or np.isnan(label)

        missing.append(is_missing)

    valid_labels = [
        label for label, is_missing in zip(labels, missing)
        if not is_missing
    ]

    try:
        distribution = Counter(valid_labels)
    except TypeError as exc:
        raise ValueError("Labels must be hashable values") from exc

    print("\n--- TerraSpectra Dataset Report ---")
    print("Total images:", image_count)
    print("Total labels:", label_count)
    print("Missing labels:", sum(missing))
    print("Disease classes:", len(distribution))
    print("\nClass distribution:")

    for disease, count in distribution.items():
        print(disease, ":", count)

    return distribution