from sdk.access import ImageClassifier
from sdk.get_max_prob import get_max_prob_desc

#sdk用法

# def predict_images(image_path, model_path='a.pth', device='cpu', batch_size=1):
# 允许指定
#  *默认模型
#  *设备类型
#  *处理图片的批量大小(实验型功能,理论上比起单队列模式可以获得更优秀的加速比,但是调用不当可能造成额外的资源消耗)
#

'''
否决方案，这个玩意速度达到了逆天的每个请求3s/的响应时间

'''
image_classifier = ImageClassifier(model_path='ResNet-0602.pth')

image_path = 'e.jpg'

max_prob_desc, max_prob = get_max_prob_desc(image_path=image_path, model_path='clip-vit-large-patch14')
print(f"Max prob description: {max_prob_desc}")
print(f"Max prob: {max_prob:.4f}")

if max_prob_desc == 'other' and max_prob > 0.95:
    print('The input content does not contain "leaf", please enter again!')
else:
    preds, idxs, label_name = image_classifier.predict_images(image_path)
    print(f'The predicted class is: {idxs}, name : {label_name}, confidence level is: {preds:.2f}')

