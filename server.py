from flask import Flask, request, jsonify
from sdk.access import predict_image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'})
    image = request.files['image']
    predicted_label = predict_image(image)
    return jsonify({'label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
