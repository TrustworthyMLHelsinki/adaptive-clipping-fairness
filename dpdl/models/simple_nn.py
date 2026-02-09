import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

class MisAlignNN(nn.Module):

    def __init__(self, input_dim=1, output_dim=10, image_size=28):
        super(MisAlignNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3)
        image_size = int((image_size - 3) / 1 + 1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        image_size = int((image_size - 3) / 1 + 1)
        self.fc = nn.Linear(16 * image_size**2, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_classifier(self):
        return self.fc

    def get_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
