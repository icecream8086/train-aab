import torch
import torchbearer
from torchbearer import Trial
from torch import nn, optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from CNN_lib.net_model import Net


# 定义数据预处理的方法
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
train_set_1 = datasets.ImageFolder('dataset', transform=transform)

basicset = torch.utils.data.ConcatDataset([train_set_1, ])
basicloader = torch.utils.data.DataLoader(basicset, batch_size=32, shuffle=True)

# 划分数据集，并打乱顺序
train_set, test_set, val_set = torch.utils.data.random_split(basicset, [0.4, 0.3, 0.3])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)




# 加载模型
model = Net()
# 加载模型状态字典
state_dict = torch.load('a.pth')

# 将状态字典加载到模型对象中
model.load_state_dict(state_dict)

# 将模型设置为评估模式
model.eval()
# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()

# 定义变量来跟踪损失和准确率
val_loss = 0
val_acc = 0
test_loss = 0
test_acc = 0

# 在验证集上评估模型
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        val_acc += (predicted == labels).sum().item()

# 在测试集上评估模型
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        test_acc += (predicted == labels).sum().item()

# 计算平均损失和准确率
val_loss /= len(val_loader.dataset)
val_acc /= len(val_loader.dataset)
test_loss /= len(test_loader.dataset)
test_acc /= len(test_loader.dataset)

print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(val_loss, val_acc))
print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc))