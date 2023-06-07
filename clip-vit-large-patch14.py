import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = "clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image_tensor = TF.to_tensor(image).to(device)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image_tensor, return_tensors="pt", padding=True)
inputs.to(device)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print(logits_per_image)
print(probs)