from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from data.volleyball_dataset import group_activities


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, len(group_activities)),
        )

        # Unfreeze the last three layers
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.avgpool.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
