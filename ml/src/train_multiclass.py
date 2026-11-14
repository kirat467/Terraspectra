from ml.src.prepare_patch_dataset import prepare_patch_dataset
from ml.src.model_3dcnn import Crop3DCNN
from ml.src.train_3dcnn import train_model, save_model

NUM_CLASSES = 5

REQUIRED_CLASSES = [
    "Healthy",
    "Alternaria alternata",
    "Alternaria solani",
    "Botrytis cinerea",
    "Fusarium oxysporum"
]


def train_multiclass_3dcnn(
    folder="ml/dataset/raw",
    epochs=10,
    batch_size=2,
    lr=0.001
):

    datasets, class_counts = prepare_patch_dataset(folder)

    missing = [
        name for name in REQUIRED_CLASSES
        if class_counts.get(name, 0) == 0
    ]

    if missing:
        raise ValueError(f"Missing classes: {missing}")

    model = Crop3DCNN(num_classes=NUM_CLASSES)

    model = train_model(
        model,
        datasets["train"],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        validation_dataset=datasets["val"]
    )

    save_model(model, "ml/models/terraspectra_3dcnn.pth")

    return model, class_counts