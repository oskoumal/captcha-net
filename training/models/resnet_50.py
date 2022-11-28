import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from training.models.image_classification_model_base import ImageClassificationModelBase


class ResNet50(ImageClassificationModelBase):
    """
        The plain ResNet-50 wrapper class - used for one captcha character and position.
    """
    def __init__(self, out_size, pretrained=False):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)

        # Replace first convolutional layer to accept greyscale image
        #self.model.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Replace the final fully connected layer to suite the problem
        self.model.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=out_size))

    def forward(self, x):
        return self.model(x)
