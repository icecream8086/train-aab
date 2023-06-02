from sdk.access import ImageClassifier

#sdk用法

# def predict_images(image_path, model_path='a.pth', device='cpu', batch_size=1):
# 允许指定
#  *默认模型
#  *设备类型
#  *处理图片的批量大小(实验型功能,理论上比起单队列模式可以获得更优秀的加速比,但是调用不当可能造成额外的资源消耗)
#


image_classifier = ImageClassifier(model_path='ResNet-0602.pth')



image_path = 'test/test_image3.jpg'
preds, idxs, label_name = image_classifier.predict_images(image_path)
print(f'The predicted class is: {idxs}, name : {label_name}, confidence level is: {preds:.2f}')


# predicted = predict_images('test/Logins_HDR.png','a.pth','cuda',1)
# print('Predicted label: {}'.format(predicted))

# # 返回
# # default mode cuda
# # Predicted label: Tomato_Leaf_Spot_Diseasepyt