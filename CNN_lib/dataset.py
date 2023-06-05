# 定义数据预处理的方法

# 定义数据处理部分
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
import cv2

def canny_edge(image, threshold1=100, threshold2=200):
    image = image.numpy().transpose(1, 2, 0) # 转换为opencv格式
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 转换为灰度图像
    image = cv2.Canny(image, threshold1, threshold2) # 进行Canny边缘检测
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # 转回RGB格式
    image = image.transpose(2, 0, 1) # 转回PyTorch格式
    return torch.from_numpy(image)


transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomRotation(10),
    # 随机擦除功能
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    transforms.ToTensor(),
    # 高亮边缘功能, 用于增强边缘特征,使用OpenCV
    transforms.Lambda(lambda image: canny_edge(image)),
    transforms.Normalize(mean=[0.485, 0.7, 0.406], std=[0.229, 0.224, 0.225])
])
