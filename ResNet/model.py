# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/17 18:43
@Author ： Rodney
@File ：model.py
@IDE ：PyCharm
"""
import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    #resnet-18和resnet-34的block
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.downsample=downsample
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        resB = x
        if self.downsample is not None:
            resB = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out+resB

        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    #resnet-50、resnet-101、resnet-152的block
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

    def forward(self, x):
        resB = x
        if self.downsample is not None:
            resB = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = out+resB
        out = self.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(Resnet,self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxplool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2 = self._make_layers(block, 64, blocks_num[0])
        self.layer3 = self._make_layers(block, 128, blocks_num[1], stride=2)
        self.layer4 = self._make_layers(block, 256, blocks_num[2], stride=2)
        self.layer5 = self._make_layers(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self,x):
        x = self.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = self.maxplool(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)
        x = self.layer5(x)
        # print(x.shape)

        if self.include_top:
            x = self.avgpool(x)
            # print(x.shape)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        # print(x.shape)
        return x



    def _make_layers(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride,bias=False),
                                       nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,channel, stride=stride,downsample=downsample))
        self.in_channel = channel*block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channel,channel))

        return nn.Sequential(*layers)

if __name__ == '__main__':
    #实例化一个resnet-18
    model = Resnet(Bottleneck, [3, 4, 6, 3])
    print(model)
    #生成一个随机的输入
    input = torch.randn(1, 3, 224, 224)
    #将输入输入到网络中
    output = model(input)
    print(output.shape)

#创建函数，返回resnet-18的实例
def resnet18(**kwargs):
    model = Resnet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

#创建函数，返回resnet-34的实例
def resnet34(**kwargs):
    model = Resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

#创建函数，返回resnet-50的实例
def resnet50(**kwargs):
    model = Resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

#创建函数，返回resnet-101的实例
def resnet101(**kwargs):
    model = Resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

#创建函数，返回resnet-152的实例
def resnet152(**kwargs):
    model = Resnet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model