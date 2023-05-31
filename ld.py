import torch
from torchbearer import Trial
from torch import nn
from torchvision import datasets
from CNN_lib.net_model import Net
from CNN_lib.dataset import transform

train_set_1 = datasets.ImageFolder('dataset', transform=transform)
basicset = torch.utils.data.ConcatDataset([train_set_1, ])
basicloader = torch.utils.data.DataLoader(basicset, batch_size=32, shuffle=True)

# 对数据集进行随机打乱
indices = torch.randperm(len(basicset))
basicset = torch.utils.data.Subset(basicset, indices)

# 划分数据集
train_size = int(len(basicset) * 0.6)
test_size = int(len(basicset) * 0.2)
val_size = len(basicset) - train_size - test_size
train_set, test_set, val_set = torch.utils.data.random_split(basicset, [train_size, test_size, val_size])

# 加载数据集
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)


# 创建模型实例
model = Net()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 定义损失函数
loss_function = nn.CrossEntropyLoss()

# 定义优化器和学习率调度器
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # 选择一个优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) # 选择一个scheduler类和参数
trial = Trial(model, optimizer, loss_function) # 创建一个trial对象，不传入 metrics 参数

# 设置训练 epoch 数和数据生成器，将数据移动到GPU
trial.for_steps(12000).with_generators(
    train_generator=train_loader,
    val_generator=val_loader,
    test_generator=test_loader
).to(device)

# 运行trial
for epoch in range(3600):
    trial.run(epochs=1) # 每次运行一个epoch
    scheduler.step() # 每个epoch后更新学习率


# 将模型移回CPU并保存
model.to('cpu')
PATH = './a.pth'
torch.save(model.state_dict(), PATH)

