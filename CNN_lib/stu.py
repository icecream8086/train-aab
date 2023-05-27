from torch import nn

class stu_net(nn.Module):
    def __init__(self, n_conv1_filters=8, n_conv2_filters=16, n_fc1_units=64, dropout_p=0.5):
        super(stu_net, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_conv1_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_conv1_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=n_conv1_filters, out_channels=n_conv2_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_conv2_filters)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 定义全连接层
        self.fc1 = nn.Linear(in_features=8*8*n_conv2_filters, out_features=n_fc1_units)
        self.bn3 = nn.BatchNorm1d(n_fc1_units)
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(in_features=n_fc1_units, out_features=10)

    def forward(self, x):
        # 前向传播
        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 8*8*self.conv2.out_channels)
        x = self.dropout(nn.functional.relu(self.bn3(self.fc1(x))))
        x = self.fc2(x)
        return x
