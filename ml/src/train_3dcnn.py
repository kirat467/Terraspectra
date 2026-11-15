import torch
import torch.nn as nn
import random
import numpy as np
import copy

from pathlib import Path
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model,
    dataset,
    epochs=10,
    batch_size=4,
    lr=0.001,
    validation_dataset=None,
    patience=3,
    checkpoint_path="ml/models/terraspectra_checkpoint.pth",
    resume=False,
    seed=42
):
    if len(dataset) == 0:
        raise ValueError("Training dataset is empty")

    if epochs <= 0 or batch_size <= 0 or lr <= 0 or patience <= 0:
        raise ValueError("Invalid training parameters")

    if validation_dataset is not None and len(validation_dataset) == 0:
        raise ValueError("Validation dataset is empty")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training device:", device)

    model = model.to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    val_loader = None

    if validation_dataset is not None:
        val_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    wait = 0
    start_epoch = 0

    checkpoint = Path(checkpoint_path)

    if resume and checkpoint.exists():
        saved = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])

        best_loss = saved["best_loss"]
        best_weights = saved["best_weights"]
        wait = saved["wait"]
        start_epoch = saved["epoch"] + 1

        print("Checkpoint loaded. Resuming training...")

    for epoch in range(start_epoch, epochs):

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Training Loss: {avg_loss:.4f} | "
            f"Training Accuracy: {train_accuracy:.2f}%"
        )

        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).long()

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()

                    predictions = outputs.argmax(dim=1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total

            print(
                f"Validation Loss: {avg_val_loss:.4f} | "
                f"Validation Accuracy: {val_accuracy:.2f}%"
            )

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_weights = copy.deepcopy(model.state_dict())
                wait = 0
                print("Best model updated!")
            else:
                wait += 1
                print(f"Early stopping patience: {wait}/{patience}")

        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            "best_weights": best_weights,
            "wait": wait,
            "seed": seed
        }, checkpoint)

        print("Checkpoint saved:", checkpoint)

        if val_loader is not None and wait >= patience:
            print("Early stopping triggered!")
            break

    if val_loader is not None:
        model.load_state_dict(best_weights)
        print("Best model weights restored!")

    return model


def save_model(model, path="ml/models/3dcnn.pth"):
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path)

    print("Model saved successfully:", save_path)