
import numpy as np
from collections import Counter


def analyze_dataset(images, labels):

    if len(images) != len(labels):
        raise ValueError("Images and labels must have equal length")

    if len(labels) == 0:
        raise ValueError("Dataset is empty")

    missing = [
        label is None or (
            isinstance(label, (float, np.floating)) and np.isnan(label)
        )
        for label in labels
    ]

    valid_labels = [
        label for label, is_missing in zip(labels, missing)
        if not is_missing
    ]

    distribution = Counter(valid_labels)

    print("\n--- TerraSpectra Dataset Report ---")
    print("Total images:", len(images))
    print("Total labels:", len(labels))
    print("Missing labels:", sum(missing))
    print("Disease classes:", len(distribution))
    print("\nClass distribution:")

    for disease, count in distribution.items():
        print(disease, ":", count)

    return distribution
