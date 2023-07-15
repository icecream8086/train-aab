import torch
import torch.nn.utils.prune as prune
from transformers import CLIPModel
from transformers import CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)

# 剪枝text_model中的token_embedding参数



prune.l1_unstructured(
    model.text_model.embeddings.token_embedding,
    name="weight",
    amount=0.5  # 设置剪枝比例
)

# 剪枝text_model中的encoder.layers参数
for layer in model.text_model.encoder.layers:
    # 获取要剪枝的权重
    weight = layer.self_attn.q_proj.weight
    # 计算要剪枝的位置
    num_prune = int(weight.shape[0] * 0.5)
    # 创建一个mask，将要剪枝的位置设为0
    mask = torch.ones(weight.shape)
    mask[:num_prune, :] = 0
    # 使用custom_from_mask函数进行剪枝
    for layer in model.text_model.encoder.layers:
        # 获取要剪枝的权重
        q_weight = layer.self_attn.q_proj.weight
        k_weight = layer.self_attn.k_proj.weight
        # 计算要剪枝的位置
        num_prune = int(q_weight.shape[0] * 0.5)
        # 创建一个mask，将要剪枝的位置设为0
        q_mask = torch.ones(q_weight.shape)
        q_mask[:num_prune, :] = 0
        k_mask = torch.ones(k_weight.shape)
        k_mask[:num_prune, :] = 0
        # 使用custom_from_mask函数进行剪枝
        prune.custom_from_mask(layer.self_attn.q_proj, name="weight", mask=q_mask.to(device))
        prune.custom_from_mask(layer.self_attn.k_proj, name="weight", mask=k_mask.to(device))

# 剪枝vision_model中的patch_embedding参数
import torch
import torch.nn.utils.prune as prune
from transformers import CLIPModel
from transformers import CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)

# Prune patch_embedding layer
prune.ln_structured(
    model.vision_model.embeddings.patch_embedding,
    name="weight",
    amount=0.5,  # Set pruning ratio
    n=None,  # Prune unstructured weights
    dim=0  # Prune along the first dimension
)

# Prune q_proj layers in encoder layers
for layer in model.vision_model.encoder.layers:
    prune.ln_structured(
        layer.self_attn.q_proj,
        name="weight",
        amount=0.5,  # Set pruning ratio
        n=1,  # Prune along one dimension
        dim=-1  # Prune along the last dimension
    )

for layer in model.text_model.encoder.layers:
    prune.ln_structured(
        layer.self_attn.q_proj,
        name="weight",
        amount=0.5,  # Set pruning ratio
        n=1,  # Prune along one dimension
        dim=-1  # Prune along the last dimension
    )

# Save pruned model and processor to a different directory
pruned_model_path = "pruned_clip_model"
model.save_pretrained(pruned_model_path)
processor.save_pretrained(pruned_model_path)


# #############################################################################################

# from transformers import CLIPModel

# model_path = "pruned_clip_model.pth"
# model = CLIPModel.from_pretrained(model_path)

# print(model)
