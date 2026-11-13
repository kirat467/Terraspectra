from pathlib import Path

from ml.src.dataset_inventory import scan_dataset
from ml.src.label_mapping import get_label
from ml.src.load_mat import load_mat_cube
from ml.src.patch_extraction import extract_spatial_patches
from ml.src.dataset import HyperspectralDataset

REQUIRED_CLASSES = [
    "Healthy",
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum"
]

def prepare_patch_dataset(
    folder="ml/dataset/raw",
    patch_size=32,
    normalization="zscore"
):
    folder = Path(folder)
    inventory = scan_dataset(folder)

    all_patches = {
        "train": [],
        "val": [],
        "test": []
    }

    all_labels = {
        "train": [],
        "val": [],
        "test": []
    }

    class_counts = {}

    for class_name in REQUIRED_CLASSES:

        files = inventory.get(class_name, [])
        class_counts[class_name] = len(files)

        if not files:
            raise ValueError(
                f"Missing dataset class: {class_name}"
            )

        for filename in files:

            cube = load_mat_cube(folder / filename)
            label = get_label(class_name)

            patches, labels = extract_spatial_patches(
                cube,
                label,
                patch_size
            )

            for split in all_patches:
                all_patches[split].extend(patches[split])
                all_labels[split].extend(labels[split])

    datasets = {}

    for split in all_patches:

        if not all_patches[split]:
            raise ValueError(f"Empty {split} dataset")

        datasets[split] = HyperspectralDataset(
            all_patches[split],
            all_labels[split],
            normalization=normalization
        )

    print("\n--- TerraSpectra Patch Dataset ---")
    print("Original samples:", sum(class_counts.values()))
    print("Patch counts:", {
        split: len(data)
        for split, data in datasets.items()
    })
    print("Classes:", class_counts)

    return datasets, class_counts