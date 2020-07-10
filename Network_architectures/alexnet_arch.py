# -*- coding: utf-8 -*-
"""
# @file name  : alexnet_arch.py
# @author     : Ares
# @date       : 2020-07-10
# @brief      : architecture of Alexnet
"""
import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self,classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 和原文相比，模型结构进行了调整，无关大雅，主要是减少了参数
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(96,256,kernel_size=3,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256,384,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(384,256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        # 全局池化，指的是一个输出通道池化为指定大小的(6,6)的特征图
        self.classifier = nn.Sequential(
            # 顺序是dropout+linear+Relu
            # 没有relu怎么做非线性拟合？
            nn.Dropout(p=0.5),
            nn.Linear(6*6*256,4096),
            # features输出了256个通道，池化为6*6的特征图
            # 输入到线性层，每个像素对应一个神经元
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096,4096),
            nn.Linear(4096,classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        # 这里必须flatten，因为分类器接受的就是6*6*256的输入
        x = self.classifier(x)
        return x