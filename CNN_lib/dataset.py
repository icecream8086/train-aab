# 定义数据预处理的方法

# 定义数据处理部分
from tkinter import Image
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

class EdgeEnhance:
    def __init__(self, kernel_size=3, blur_radius=3):
        self.kernel_size = kernel_size
        self.blur_radius = blur_radius
    
    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (self.blur_radius, self.blur_radius), 0)
        img_laplacian = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=self.kernel_size)
        img_laplacian = cv2.cvtColor(img_laplacian, cv2.COLOR_GRAY2RGB)
        return img_laplacian
    
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=7),  # 添加高斯平滑处理
    EdgeEnhance(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.6, 0.406], std=[0.229, 0.224, 0.225])
])

# transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.6, 0.406], std=[0.229, 0.224, 0.225])
# ])