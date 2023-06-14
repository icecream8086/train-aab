import torch
from torchvision.transforms import functional as TF
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 在函数外部初始化模型和处理器
model_path = "clip-vit-base-patch32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)

def get_max_prob_desc(image_path, texts=["leaf", "other"], model=model, processor=processor):
    print(f"Using device: {device}\nModel: {model_path}")

    image = Image.open(image_path)
    image_tensor = TF.to_tensor(image).to(device)

    inputs = processor(text=texts, images=image_tensor, return_tensors="pt", padding=True)
    inputs.to(device)

    with torch.no_grad():  # 禁用梯度计算
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    max_prob_idx = probs.argmax(-1).item()
    max_prob_desc = texts[max_prob_idx]

    return max_prob_desc, probs[0][max_prob_idx].item()

