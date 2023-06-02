import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes, freeze_layers=True):
        super(ResNet50, self).__init__()
        self.num_classes = num_classes
        self.resnet50 = models.resnet50(pretrained=True)
        if freeze_layers:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)

        x = self.resnet50.layer1(x)
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)

        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x