import torch
from torch.utils.data import Dataset

try:
    from .normalization import normalize_spectral_cube
except ImportError:
    from normalization import normalize_spectral_cube


class HyperspectralDataset(Dataset):

    def __init__(self, images, labels, normalization=None):
        self.images = images
        self.labels = labels
        self.normalization = normalization

        if len(images) != len(labels):
            raise ValueError("Images and labels must match")

        if len(images) == 0:
            raise ValueError("Dataset is empty")

        if normalization not in (None, "zscore", "minmax", "snv"):
            raise ValueError("Invalid normalization method")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]

        if self.normalization:
            image = normalize_spectral_cube(
                image, method=self.normalization
            )

        image = torch.as_tensor(
            image, dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[index], dtype=torch.long
        )

        # Convert H, W, Bands into 1, Bands, H, W
        image = image.permute(2, 0, 1).unsqueeze(0)

        return image, label