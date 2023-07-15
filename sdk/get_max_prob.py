import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import torchvision.transforms.functional as TF

def get_max_prob_desc(image_path, texts=["leaf", "other"], model_path="clip-vit-large-patch14"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} \n model: {model_path}")
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)

    image = Image.open(image_path)
    image_tensor = TF.to_tensor(image).to(device)

    inputs = processor(text=texts, images=image_tensor, return_tensors="pt", padding=True)
    inputs.to(device)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    max_prob_idx = probs.argmax(-1).item()
    max_prob_desc = texts[max_prob_idx]
    
    return max_prob_desc, probs[0][max_prob_idx].item()
