import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path


def train_model(
    model,
    dataset,
    epochs=10,
    batch_size=4,
    lr=0.001,
    validation_dataset=None
):
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")

    if epochs <= 0 or batch_size <= 0 or lr <= 0:
        raise ValueError("Invalid training parameters")

    if validation_dataset is not None and len(validation_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    model = model.to(device)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = None

    if validation_dataset is not None:
        val_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Training Loss: {avg_loss:.4f}"
        )

        if val_loader is not None:

            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(device)
                    labels = labels.to(device).long()

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    predictions = outputs.argmax(dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total

            print(
                f"Validation Loss: {avg_val_loss:.4f} "
                f"| Validation Accuracy: {accuracy:.2f}%"
            )

    return model


def save_model(model, path="ml/models/3dcnn.pth"):

    save_path = Path(path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(model.state_dict(), save_path)

    print("Model saved successfully:", save_path)