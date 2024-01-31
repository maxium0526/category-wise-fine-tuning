import torch
from torchvision.models import resnet101
from torch import nn

class Model(torch.nn.Module):
    def __init__(self, num_categories):
        super().__init__()

        self.backbone = resnet101(weights='IMAGENET1K_V2')
        self.backbone.fc = nn.Identity()
        self.classification_layer = nn.Linear(2048, num_categories)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classification_layer(x)
        return x