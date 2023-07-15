from flask import Flask, request, jsonify
from sdk.access import ImageClassifier
import os
import uuid
from sdk.get_max_prob import get_max_prob_desc  # 导入 get_max_prob_desc 函数
from PIL import Image
import io

app = Flask(__name__)

# 设置上传文件的最大大小为 16 MB
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# 初始化图像分类器
image_classifier = ImageClassifier(model_path='ResNet-0602.pth')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # 将图片文件保存到 uploads 目录下，使用安全的文件名
    filename = str(uuid.uuid4()) + '.jpg'
    image_path = os.path.join('uploads', filename)
    file.save(image_path)

    # 裁剪图片为 256x256
    with Image.open(image_path) as img:
        img = img.resize((256, 256))
        img.save(image_path)

    # 获取概率最大的标签及其概率
    max_prob_desc, max_prob = get_max_prob_desc(image_path=image_path, model_path='clip-vit-base-patch32')
    print(f"Max prob description: {max_prob_desc}")
    print(f"Max prob: {max_prob:.4f}")

    if max_prob_desc == 'leaf' and max_prob > 0.95:  # 如果概率最大的标签是 'leaf'，执行图像分类器
        preds, idxs, label_name = image_classifier.predict_images(image_path)
        result = {
            'predicted_class': idxs ,
            'class_name': label_name,
            'confidence_level': preds,
            'file_path': image_path,
            'file_name': file.filename 
        }
    else:  # 如果概率最大的标签是 'leave'，返回错误信息
        result = {
            'error': 'The input content does not contain "leaf", please enter again!'
        }
    return jsonify(result)

if __name__ == '__main__':
    # 创建 uploads 目录
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0')
