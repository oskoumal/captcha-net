import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.image_classification_model_base import ImageClassificationModelBase


class MLP(ImageClassificationModelBase):
    """
        Feedforward neural network with 1 hidden layer.
    """
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)

        self.running_loss = True

    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
