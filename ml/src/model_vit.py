
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

        self.embedding = nn.Linear(input_dim, embed_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.classifier(x)
