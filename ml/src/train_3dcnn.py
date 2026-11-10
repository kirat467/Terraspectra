
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model, dataset, epochs=10, batch_size=4, lr=0.001):

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
        model.parameters(), lr=lr
    )

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Loss: {total_loss/len(loader):.4f}"
        )

    return model


def save_model(model, path="ml/models/3dcnn.pth"):
    torch.save(model.state_dict(), path)
    print("Model saved successfully!")
