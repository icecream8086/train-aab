from sdk.access import predict_image

#sdk用法
predicted = predict_image('tomato.jpg')
print('Predicted label: {}'.format(predicted))

# # 返回
# # mode cuda
# # Predicted label: Tomato_Leaf_Spot_Disease