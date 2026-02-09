import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)

    def get_classifier(self):
        return self.linear

    def get_transforms(self):
        return transforms.Compose([])
