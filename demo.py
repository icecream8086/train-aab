from sdk.access import predict_image,predict_images

#sdk用法

# 简化版
# predict_image('test_image.jpg')
predicted = predict_image('test_image.jpg')
print('Predicted label: {}'.format(predicted))

# # 返回
# # default mode cuda
# # Predicted label: Tomato_Leaf_Spot_Disease


# 复杂版 
#
# def predict_images(image_path, model_path='a.pth', device='cpu', batch_size=1):
# 允许指定
#  *默认模型
#  *设备类型
#  *处理图片的批量大小(实验型功能,理论上比起单队列模式可以获得更优秀的加速比,但是调用不当可能造成额外的资源消耗)
#

predicted = predict_images('apple.jpg','a.pth','cpu',1)
print('Predicted label: {}'.format(predicted))

# # 返回
# # default mode cuda
# # Predicted label: Tomato_Leaf_Spot_Disease