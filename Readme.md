# 农业病害识别项目

## 简介

此项目使用卷积神经网络实现了对农业作物病害的识别。

模型基于一个基本的CNN,根据具体任务需要进行了一些调整和优化，针对卷积层和全连接层的输出节点数、dropout的概率、BatchNormalization的位置等进行了调整，调参时使用torchbearer进行自适应权重衰减以达到更好的性能。

[点击](CNN_lib/note.md)查看

## 模型

此项目使用了自定义的卷积神经网络模型，包含两个卷积层和两个全连接层。

输入是图片，经过卷积、池化和全连接等操作，最终输出10个节点，对应10分类的输出。

同时，模型中添加了批归一化和dropout层，以加强模型的稳定性和防止过拟合。

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
