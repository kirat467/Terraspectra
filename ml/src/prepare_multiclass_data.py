from pathlib import Path

from ml.src.dataset_inventory import scan_dataset
from ml.src.label_mapping import get_label
from ml.src.load_mat import load_mat_cube
from ml.src.dataset import HyperspectralDataset


def prepare_multiclass_data(
    folder="ml/dataset/raw",
    normalization="zscore"
):
    folder = Path(folder)

    inventory = scan_dataset(folder)

    images = []
    labels = []
    class_counts = {}

    for class_name, files in inventory.items():

        class_counts[class_name] = len(files)

        for filename in files:

            file_path = folder / filename

            cube = load_mat_cube(file_path)

            images.append(cube)
            labels.append(get_label(class_name))

    if not images:
        raise ValueError("No valid dataset samples found")

    dataset = HyperspectralDataset(
        images,
        labels,
        normalization=normalization
    )

    print("\n--- Multi-class Data Preparation ---")
    print("Total samples:", len(dataset))
    print("Class counts:", class_counts)
    print("Normalization:", normalization)

    return dataset, class_counts