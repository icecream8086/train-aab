import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from CNN_lib.dataset_sample import transform


# 读取图片
img_path = 'apple.jpg'
img = Image.open(img_path)

img_processed = transform(img)

# 可视化处理后的图像
img_processed_np = img_processed.numpy().transpose((1, 2, 0))
plt.imshow(img_processed_np)
plt.show()

# # 保存处理后的图像
# img_processed_pil = transforms.ToPILImage()(img_processed)
# save_path = 'apple-normal.jpg'
# img_processed_pil.save(save_path)

# 保存处理后的图像
img_processed_pil = transforms.ToPILImage()(img_processed)
save_path = 'apple-plus2.jpg'
img_processed_pil.save(save_path)
