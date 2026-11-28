import torch
import torch.nn as nn


class CropViT(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes,
        embed_dim=128,
        num_heads=4,
        num_layers=2
    ):
        super(CropViT, self).__init__()

        # Validate constructor parameters
        if input_dim <= 0:
            raise ValueError("Input dimension must be positive")

        if num_classes <= 0:
            raise ValueError("Number of classes must be positive")

        if embed_dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        if num_heads <= 0:
            raise ValueError("Number of attention heads must be positive")

        if num_layers <= 0:
            raise ValueError("Number of transformer layers must be positive")

        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads"
            )

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Feature embedding
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Classification layer
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        # Validate input tensor
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input must be a PyTorch tensor")

        # Validate dimensions
        if x.ndim != 3:
            raise ValueError(
                "ViT input must be 3D: "
                "[batch, sequence_length, input_dim]"
            )

        # Validate batch
        if x.shape[0] == 0:
            raise ValueError("Input batch cannot be empty")

        # Validate sequence length
        if x.shape[1] == 0:
            raise ValueError("Sequence length cannot be empty")

        # Validate feature dimension
        if x.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features, "
                f"received {x.shape[2]}"
            )

        # Feature embedding
        x = self.embedding(x)

        # Transformer processing
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification
        return self.classifier(x)