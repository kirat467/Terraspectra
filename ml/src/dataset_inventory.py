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

    files = list(folder.glob("*.mat"))
    inventory = {name: [] for name in CLASS_NAMES.values()}

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
        name for name, samples in inventory.items()
        if not samples
    ]

    print("\nTotal samples:", sum(map(len, inventory.values())))
    print("Missing classes:", missing)

    return inventory