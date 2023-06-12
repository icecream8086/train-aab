import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision.transforms import transforms
from CNN_lib.net_model import ResNet_0602
from CNN_lib.dataset_sample import transform

class ImageClassifier:
    def __init__(self, model_path='ResNet-0602.pth'):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device} \n model: {model_path}')
        self.model = ResNet_0602(num_classes=10)  # TODO: 修改 num_classes 为具体的分类数目
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(device=self.device)
        self.model.eval()

        self.transform = transform

    def predict_images(self, image_path, batch_size=1):
        images = self._load_images(image_path)
        with torch.no_grad():
            logits = self.model(images.to(device=self.device))
            preds = torch.softmax(logits, dim=-1)
            preds, idxs = torch.topk(preds, k=1, dim=-1)
            preds, idxs = preds.item(), idxs.item()

            # 获取对应的标签名字
            label_names = ['Apple_Black_Rot_Disease', 'Grape_Black_Rot_Disease', 'Tomato_Leaf_Spot_Disease', ...]  # TODO:修改为具体的标签名字
            label_name = label_names[idxs]

        return preds, idxs, label_name

    def _load_images(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)
