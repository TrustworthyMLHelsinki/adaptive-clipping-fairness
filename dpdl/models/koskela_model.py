import logging
import torch
import torch.nn.functional as F

from torch import nn
from torchvision import transforms

log = logging.getLogger(__name__)

# for grayscale images
class KoskelaNetGrayscale(nn.Module):
    """
    This is the network used in the paper "Learning Rate Adaptation for Federated and
    Differentially Private Learning" (Koskela et al., 2019)

    https://arxiv.org/pdf/1809.03832.pdf
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        self.final_fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final_fc(x)

        return x

    def get_classifier(self):
        return self.final_fc

    def get_transforms(self):
        return transforms.Compose(
            [
                # transforms.ToTensor(),
            ]
        )
