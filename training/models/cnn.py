import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.image_classification_model_base import ImageClassificationModelBase


class CNN(ImageClassificationModelBase):
    """
        Slightly involved CNN neural network.
    """
    def __init__(self, out_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Flatten(),
            nn.Linear(in_features=24200, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=out_size, bias=True)
        )

    def forward(self, xb):
        return self.network(xb)
