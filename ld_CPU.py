import torch
import torchbearer
from torchbearer import Trial
from torch import nn, optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


# 定义数据预处理的方法
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_set_1 = datasets.ImageFolder('dataset', transform=transform)

# train_set_2 = datasets.ImageFolder('dataset/Grape_Black_Rot_Disease', transform=transform)
# train_set_3 = datasets.ImageFolder('dataset/Tomato_Leaf_Spot_Disease', transform=transform)

# 将三个数据集合并成一个数据集，并打乱顺序
basicset = torch.utils.data.ConcatDataset([train_set_1, ])
basicloader = torch.utils.data.DataLoader(basicset, batch_size=32, shuffle=True)

# 划分数据集，并打乱顺序
train_set, test_set, val_set = torch.utils.data.random_split(basicset, [0.6, 0.2, 0.2])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)


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


# 定义参数字典
params = {
    "n_conv1_filters": [6, 12, 18],
    "n_conv2_filters": [16, 32, 48],
    "n_fc1_units": [120, 240, 360],
    "n_fc2_units": [84, 168, 252]
}

# 创建模型实例
model = Net()

# 定义损失函数
loss_function = nn.CrossEntropyLoss()

# 使用torch.optim.lr_scheduler进行学习率调整
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # 选择一个优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) # 选择一个scheduler类和参数
trial = Trial(model, optimizer, loss_function, metrics=['loss']) # 创建一个trial对象，不需要传入scheduler作为回调函数

# 设置训练步数和数据生成器
trial.for_steps(100).with_generators(
    train_generator=train_loader,
    val_generator=val_loader,
    test_generator=test_loader
).run(epochs=10)

# 运行trial
for epoch in range(10):
    trial.run(epochs=1) # 每次运行一个epoch
    scheduler.step() # 每个epoch后更新学习率



# 保存模型
PATH = './a.pth'
torch.save(model.state_dict(), PATH)
