import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from CNN_lib.net_model import ResNet_0602

'''
修改Res-Net模型的剪枝代码，使它能够被正确地加载

'''
model = ResNet_0602(num_classes=10, freeze_layers=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("ResNet-0602.pth"))
model.to(device)

prune_percentage = 0.01  # Pruning ratio, adjust as needed

# Prune convolutional layers in the resnet50 module
for module in model.resnet50.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=prune_percentage)

# Prune linear layers in the fc module
for module in model.fc.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=prune_percentage)

# Remove pruning-related parameters
prune.remove(model.resnet50.conv1, 'weight')
prune.remove(model.fc[3], 'weight')

# Create a new state dictionary for the pruned model
pruned_state_dict = model.state_dict()

# Update the keys in the pruned state dictionary
for key in list(pruned_state_dict.keys()):
    new_key = key.replace("resnet50.", "")
    if new_key in pruned_state_dict:
        pruned_state_dict[new_key] = pruned_state_dict.pop(key)

# Load the modified state dictionary into the model
model.load_state_dict(pruned_state_dict)

# Save the pruned model
torch.save(model.state_dict(), "ResNet-0602-pruned.pth")
print("Model saved to ResNet-0602-pruned.pth")