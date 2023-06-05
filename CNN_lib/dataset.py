# 定义数据预处理的方法

# 定义数据处理部分
from tkinter import Image
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

class EdgeEnhance:
    def __call__(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.Laplacian(img, cv2.CV_32F)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(np.uint8(img))
        img_gray_pil = img_pil.convert('L')
        img_gray_rgb = img_gray_pil.convert('RGB')
        return img_gray_rgb
    
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    EdgeEnhance(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.7, 0.406], std=[0.229, 0.224, 0.225])
])
