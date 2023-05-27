from torch import nn

class Net(nn.Module):
    def __init__(self, n_conv1_filters=6, n_conv2_filters=16, n_fc1_units=120, n_fc2_units=84):
        super(Net, self).__init__()
        # 定义卷积层和池化层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_conv1_filters, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=n_conv1_filters, out_channels=n_conv2_filters, kernel_size=5)
        # 定义全连接层
        self.fc1 = nn.Linear(in_features=44944, out_features=n_fc1_units) 
        self.fc2 = nn.Linear(in_features=n_fc1_units, out_features=n_fc2_units) 
        self.fc3 = nn.Linear(in_features=n_fc2_units, out_features=10)

    def forward(self, x):
        # 前向传播
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除了batch size的所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

