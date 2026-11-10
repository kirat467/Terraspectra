
import torch
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        if len(images) != len(labels):
            raise ValueError("Images and labels must match")

        if len(images) == 0:
            raise ValueError("Dataset is empty")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = torch.as_tensor(
            self.images[index],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        # Convert H, W, Bands into 1, Bands, H, W
        image = image.permute(2, 0, 1).unsqueeze(0)

        return image, label

