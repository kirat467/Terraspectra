from ml.src.prepare_multiclass_data import prepare_multiclass_data
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
    batch_size=4,
    lr=0.001
):

    dataset, class_counts = prepare_multiclass_data(folder)

    missing = [
        name for name in REQUIRED_CLASSES
        if class_counts.get(name, 0) == 0
    ]

    if missing:
        raise ValueError(
            f"Missing disease classes: {missing}"
        )

    model = Crop3DCNN(num_classes=NUM_CLASSES)

    model = train_model(
        model,
        dataset,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )

    save_model(model, "ml/models/terraspectra_3dcnn.pth")

    return model, class_counts