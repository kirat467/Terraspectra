import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import math


def train_vit(model, dataset, epochs=10, batch_size=4, lr=0.001):

    # Validate dataset
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")

    # Validate training parameters
    if (
        not isinstance(epochs, int)
        or isinstance(epochs, bool)
        or epochs <= 0
    ):
        raise ValueError("Invalid training parameters")

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("Invalid training parameters")

    if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid training parameters")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0
        total_samples = 0

        for images, labels in loader:

            # Validate batch tensors
            if not isinstance(images, torch.Tensor):
                raise ValueError("Training images must be tensors")

            if not isinstance(labels, torch.Tensor):
                raise ValueError("Training labels must be tensors")

            # Validate image dimensions
            if images.ndim != 3:
                raise ValueError(
                    "ViT training images must be 3D: "
                    "[batch, sequence_length, input_dim]"
                )

            if images.shape[0] == 0:
                raise ValueError("Training batch cannot be empty")

            if images.shape[1] == 0 or images.shape[2] == 0:
                raise ValueError("Training image dimensions cannot be empty")

            # Validate finite image values
            if not torch.isfinite(images).all():
                raise ValueError("Training images must contain finite values")

            # Validate labels
            if labels.ndim != 1:
                raise ValueError("Training labels must be 1D")

            if labels.shape[0] != images.shape[0]:
                raise ValueError("Image and label batch sizes must match")

            if labels.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8
            ):
                raise ValueError("Training labels must contain integers")

            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(images)

            # Validate model outputs
            if outputs.ndim != 2:
                raise ValueError("Model must return 2D class logits")

            if outputs.shape[0] != labels.shape[0]:
                raise ValueError("Model output and label batch sizes must match")

            if outputs.shape[1] == 0:
                raise ValueError("Model must return at least one class")

            if not torch.isfinite(outputs).all():
                raise ValueError("Model outputs must contain finite values")

            if torch.any(labels < 0) or torch.any(
                labels >= outputs.shape[1]
            ):
                raise ValueError("Training labels contain invalid class indices")

            loss = criterion(outputs, labels)

            if not torch.isfinite(loss):
                raise ValueError("Training loss must be finite")

            loss.backward()
            optimizer.step()

            batch_count = images.shape[0]

            total_loss += loss.item() * batch_count
            total_samples += batch_count

        avg_loss = total_loss / total_samples

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Loss: {avg_loss:.4f}"
        )

    return model


def save_vit(model, path="ml/models/vit.pth"):

    save_path = Path(path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(model.state_dict(), save_path)

    print("ViT model saved successfully:", save_path)