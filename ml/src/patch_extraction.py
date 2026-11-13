import numpy as np


def extract_spatial_patches(
    cube,
    label,
    patch_size=32
):
    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise ValueError("Cube must be 3-dimensional")

    if not np.isfinite(cube).all():
        raise ValueError("Cube contains invalid values")

    height, width, bands = cube.shape

    if height < patch_size or width < patch_size:
        raise ValueError("Image is too small")

    patches = {
        "train": [],
        "val": [],
        "test": []
    }

    labels = {
        "train": [],
        "val": [],
        "test": []
    }

    for row in range(0, height - patch_size + 1, patch_size):

        for col in range(0, width - patch_size + 1, patch_size):

            column_index = col // patch_size

            # Leave buffer columns between splits
            if column_index == 4 or column_index == 6:
                continue

            patch = cube[
                row:row + patch_size,
                col:col + patch_size,
                :
            ]

            if column_index <= 3:
                split = "train"
            elif column_index == 5:
                split = "val"
            else:
                split = "test"

            patches[split].append(
                patch.astype(np.float32)
            )

            labels[split].append(label)

    return patches, labels