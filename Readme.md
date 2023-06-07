# 农业病害识别项目

## 简介

此项目使用卷积神经网络实现了对农业作物病害的识别。

模型基于一个基本的CNN,根据具体任务需要进行了一些调整和优化，针对卷积层和全连接层的输出节点数、dropout的概率、BatchNormalization的位置等进行了调整，调参时使用torchbearer进行自适应权重衰减以达到更好的性能。

[点击](CNN_lib/note.md)查看

## 模型

> #TODO: 针对clip-vit-base-patch16进行剪枝和蒸馏，clip-vit-large-patch14 太慢了

基于ResNet50模型进行预训练，如果想添加是否包含树叶的功能，此模型对于数据据之外的内容无能为力，可以加clip-vit-large-patch14模型先进行预测处理，再使用此模型进行预测。但是,这样可能会消耗更多的资源

关于clip-vit-large-patch14
> clip-vit-large-patch14 下载地址为 http://10.21.156.183:8080/ [账户:src,密码:1] (如果无法打开URL,应该是因为服务器离线，请联系我)

## 数据预处理

在加载数据集时，使用了对数据进行resize、裁剪、归一化等操作的预处理方法。

## 训练

训练过程中使用了SGD优化器和学习率衰减策略，通过10000个epoch的训练

测试集准确率高达 98%

```shell

# Validation Loss: 0.0006, Validation Accuracy: 0.9944

# Test Loss: 0.0016, Test Accuracy: 0.9902


```

## 关于内容

| 文件名               | 作用               | 注释                                 |
| -------------------- | ------------------ | ------------------------------------ |
| unit_test.py         | 神经网络准确度评估 | 训练集 测试集 验证集比例分别为 6:2:2 |
| ld.py                | 神经网络训练       | 调参为自动调参                       |
| distiller.py         | 模型蒸馏           | 未完成                               |
| demo.py              | 演示如何使用       |                                      |
| sdk/access.py        | 快速调用方法接口   |                                      |
| Log/statusLog.py     | 日志功能           | 未使用                               |
| CNN_lib/net_model.py | 神经网络核心定义   | CNN网络的基本定义                             |

## 版权声明

> 此项目为小组核心机密，未经允许不得外传。
