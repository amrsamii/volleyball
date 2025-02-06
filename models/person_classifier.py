from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class PersonClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = resnet.fc.in_features

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(num_features, 265),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(265, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
