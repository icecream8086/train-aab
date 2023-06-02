from flask import Flask, request, jsonify
from sdk.access import ImageClassifier
import os
import uuid

app = Flask(__name__)

image_classifier = ImageClassifier(model_path='ResNet-0602.pth')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # 将图片文件保存到 uploads 目录下，使用安全的文件名
    filename = str(uuid.uuid4()) + '.jpg'
    image_path = os.path.join('uploads', filename)
    file.save(image_path)

    preds, idxs, label_name = image_classifier.predict_images(image_path)
    result = {
        'predicted_class': idxs ,
        'class_name': label_name,
        'confidence_level': preds
    }
    return jsonify(result)

if __name__ == '__main__':
    # 创建 uploads 目录
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True,host='0.0.0.0')
