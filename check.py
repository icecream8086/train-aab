import torch
from CNN_lib.net_model import ResNet_0602

model = ResNet_0602(num_classes=5)
input = torch.randn(1, 3, 224, 224)
output = model(input)

print(f'Input shape: {input.shape}')
print(f'Output shape: {output.shape}')
