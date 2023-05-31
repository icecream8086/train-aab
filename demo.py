from sdk.access import predict_images

#sdk用法

# def predict_images(image_path, model_path='a.pth', device='cpu', batch_size=1):
# 允许指定
#  *默认模型
#  *设备类型
#  *处理图片的批量大小(实验型功能,理论上比起单队列模式可以获得更优秀的加速比,但是调用不当可能造成额外的资源消耗)
#

predicted = predict_images('test/test_image0.jpg','a.pth','cuda',1)
print('Predicted label: {}'.format(predicted))
predicted = predict_images('test/test_image1.jpg','a.pth','cuda',1)
print('Predicted label: {}'.format(predicted))
predicted = predict_images('test/test_image2.jpg','a.pth','cuda',1)
print('Predicted label: {}'.format(predicted))
predicted = predict_images('test/test_image3.jpg','a.pth','cuda',1)
print('Predicted label: {}'.format(predicted))
predicted = predict_images('test/test_image4.jpg','a.pth','cuda',1)
print('Predicted label: {}'.format(predicted))
# predicted = predict_images('test/Logins_HDR.png','a.pth','cuda',1)
# print('Predicted label: {}'.format(predicted))

# # 返回
# # default mode cuda
# # Predicted label: Tomato_Leaf_Spot_Disease