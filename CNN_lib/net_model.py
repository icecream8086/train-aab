import torch
import torch.nn as nn
import torchvision.models as models

class ResNet_0602(nn.Module):
    def __init__(self, num_classes, freeze_layers=True):
        super(ResNet_0602, self).__init__()
        self.num_classes = num_classes
        self.resnet50 = models.resnet50(pretrained=True)
        if freeze_layers:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        
        self.resnet50.fc = nn.Identity()  # 使用Identity代替ReLU(inplace=True)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
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



class Cnn_Net(nn.Module):
    def __init__(self, n_conv1_filters=6, n_conv2_filters=16, n_fc1_units=120, n_fc2_units=84):
        super(Cnn_Net, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_conv1_filters, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(n_conv1_filters)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=n_conv1_filters, out_channels=n_conv2_filters, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(n_conv2_filters)
        # 定义全连接层
        self.fc1 = nn.Linear(in_features=44944, out_features=n_fc1_units)
        self.bn3 = nn.BatchNorm1d(n_fc1_units)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=n_fc1_units, out_features=n_fc2_units)
        self.bn4 = nn.BatchNorm1d(n_fc2_units)
        self.fc3 = nn.Linear(in_features=n_fc2_units, out_features=10)

    def forward(self, x):
        # 前向传播
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(nn.functional.relu(self.bn3(self.fc1(x))))
        x = self.dropout(nn.functional.relu(self.bn4(self.fc2(x))))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除了batch size的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

'''
# 警告,此描述为GitHub Copilot自动生成的描述,可能不准确 更新日期:2023年6月14日 15点22分
GitHub Copilot：这是一个使用PyTorch库定义神经网络模型的Python脚本。该模型名为ResNet_0602，是nn.Module类的子类。ResNet_0602模型接受一个输入张量，并返回一个大小为num_classes的输出张量。

ResNet_0602模型基于ResNet50架构，它是一个在ImageNet数据集上进行预训练的深度卷积神经网络。预训练的ResNet50模型使用torchvision.models.resnet50函数加载。

ResNet50模型有几个层，包括卷积层、批归一化层、ReLU激活层和全连接层。ResNet_0602模型用一个恒等层替换了ResNet50模型的最后一个全连接层，这个层只是简单地将输入张量传递给输出张量。恒等层的输出然后通过一个具有ReLU激活函数的全连接层、一个dropout层和另一个全连接层进行处理，以产生最终的输出张量。

ResNet_0602模型还有一个选项，可以通过将freeze_layers参数设置为True来在训练期间冻结ResNet50层的权重。当在新数据集上微调模型时，这可能很有用，因为它允许模型学习新的特征，而不会覆盖在ImageNet数据集上学习的预训练特征。

要使用ResNet_0602模型，您可以创建类的一个实例，并将数据集中的类数作为num_classes参数传递。然后，您可以使用输入张量调用模型的forward方法以获取输出张量。

提高代码可读性和性能的可能方法包括添加注释以解释每个层和参数的目的，使用更具描述性的变量名称，并尝试使用不同的激活函数和dropout率来提高模型的准确性。此外，将代码拆分为较小的函数或模块可能很有用，使其更模块化，更易于测试和维护。'''