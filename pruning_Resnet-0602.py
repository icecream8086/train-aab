import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from CNN_lib.net_model import ResNet_0602

'''

'''
model = ResNet_0602(num_classes=10, freeze_layers=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("ResNet-0602.pth"))
model.to(device)

prune_percentage = 0.5  # 剪枝比例，根据需求进行调整

# 针对resnet50模型中的卷积层进行剪枝
for module in model.resnet50.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=prune_percentage)

# 针对线性层进行剪枝
for module in model.fc.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=prune_percentage)

prune.remove(model.resnet50.conv1, 'weight')  # 删除剪枝辅助参数
prune.remove(model.fc[3], 'weight')  # 删除剪枝辅助参数

torch.save(model, "ResNet-0602-pruned.pth")
