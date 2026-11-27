from pathlib import Path


CLASS_NAMES = {
    "healthy": "Healthy",
    "alternaria_alternata": "Alternaria alternata",
    "alternaria_solani": "Alternaria solani",
    "botrytis_cinerea": "Botrytis cinerea",
    "fusarium_oxysporum": "Fusarium oxysporum"
}


def scan_dataset(folder="ml/dataset/raw"):

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError("Dataset folder not found")

    if not folder.is_dir():
        raise NotADirectoryError("Dataset path must be a directory")

    inventory = {
        class_name: []
        for class_name in CLASS_NAMES.values()
    }

    try:
        files = sorted(
            [
                file for file in folder.iterdir()
                if file.is_file() and file.suffix.lower() == ".mat"
            ],
            key=lambda file: file.name.lower()
        )

    except PermissionError as error:
        raise PermissionError(
            "Permission denied while accessing dataset folder"
        ) from error

    for file in files:

        filename = file.stem.lower()

        for key, class_name in CLASS_NAMES.items():

            if key in filename:
                inventory[class_name].append(file.name)
                break

    print("\n--- TerraSpectra Dataset Inventory ---")

    for class_name, samples in inventory.items():
        print(f"{class_name}: {len(samples)} samples")

    missing = [
        name
        for name, samples in inventory.items()
        if not samples
    ]

    total_samples = sum(
        len(samples)
        for samples in inventory.values()
    )

    print("\nTotal samples:", total_samples)
    print("Missing classes:", missing)

    return inventory