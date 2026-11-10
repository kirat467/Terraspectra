
import torch
import torch.nn as nn

class Crop3DCNN(nn.Module):

    def __init__(self, num_classes):
        super(Crop3DCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.AdaptiveAvgPool3d(1)
        )

        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
