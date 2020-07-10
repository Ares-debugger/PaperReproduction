# -*- coding: utf-8 -*-
"""
# @file name  : VGG_archs.py
# @author     : Ares
# @date       : 2020-07-10
# @brief      : the advanced way to create a VGG Net
"""
import torch.nn as nn
import torch

__all__ = ['VGG','vgg16','vgg19','vgg16_bn','vgg19_bn']
# 标识了这个模块中的哪些属性可以被导入到其他模块当中

class VGG(nn.Module):
    # 只保留通用的部分，features需要根据cfg构造，自顶向下编程思想
    def __init__(self,features,classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(6*6*512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

cfgs = {
    '16_layer':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    '19_layer':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def _vgg(cfg,batch_norm):
    model = VGG(make_layers(cfgs[cfg],batch_norm)) # 传入构造好的features
    return model

def make_layers(cfg,batch_norm=False):
    '''
    :param cfg: 列表
    :param batch_norm: 是否进行BN
    :return:model
    变化的初始值思想！！！变化的初始值思想！！
    '''
    layers = []
    in_channels = 3
    for value in cfg:
        if value == 'M':
            # 这里不能append，这里只能+=，因为每一个都是单个进入列表的
            layers += nn.MaxPool2d(kernel_size=2,stride=2)
        else:
            conv2d = nn.Conv2d(in_channels,value,kernel_size=3,stride=1,padding=1)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(value),nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,nn.ReLU(inplace=True)]
            in_channels = value # 修改in_channels 为下一层使用的变化初始值思想
    return nn.Sequential(*layers) # 列表元素拆开使用

def vgg16():
    return _vgg('16_layer',batch_norm=False)

def vgg19():
    return _vgg('16_layer',batch_norm=False)

def vgg16_bn():
    return _vgg('19_layer',batch_norm=True)

def vgg19_bn():
    return _vgg('19_layer',batch_norm=True)


