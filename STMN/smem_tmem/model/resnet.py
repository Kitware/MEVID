import os
import math
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torchvision import models

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck_key(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_key, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out

class ResNet(nn.Module):
    def __init__(self, last_stride=1, block=Bottleneck, last_block=Bottleneck_key, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_val = self._make_layer(block, 512, layers[3], stride=last_stride)
        self.inplanes = 1024
        self.layer4_key_s = self._make_layer_key(block, last_block, 512, layers[3], stride=last_stride)
        self.inplanes = 1024
        self.layer4_key_t = self._make_layer_key(block, last_block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_layer_key(self, block, last_block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(last_block(self.inplanes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # branch
        val = self.layer4_val(x)
        key_s = self.layer4_key_s(x)
        key_t = self.layer4_key_t(x)
        return val, key_s, key_t

class Resnet50(nn.Module):
    def __init__(self, pooling=True, stride=1):
        super(Resnet50, self).__init__()
        original = models.resnet50(pretrained=True).state_dict()
        self.backbone = ResNet(last_stride=stride)

        cnt = 0
        layer4_val = self.get_key('layer4_val')
        layer4_key_s = self.get_key('layer4_key_s')
        layer4_key_t = self.get_key('layer4_key_t')
        for key in original:
            if key.find('fc') != -1:
                continue
            else:
                if key.find('layer4') != -1:
                    self.backbone.state_dict()[layer4_val[cnt]].copy_(original[key])
                    self.backbone.state_dict()[layer4_key_s[cnt]].copy_(original[key])
                    self.backbone.state_dict()[layer4_key_t[cnt]].copy_(original[key])
                    cnt+=1
                else:
                    self.backbone.state_dict()[key].copy_(original[key])
        del original

        if pooling == True:
            self.add_module('avgpool', nn.AdaptiveAvgPool2d(1))
        else:
            self.avgpool = None
        self.out_dim = 2048

    def get_key(self, layer):
        out = []
        for key in self.backbone.state_dict():
            if key.find(layer) != -1:
                out.append(key)
        return out

    def forward(self, x):
        val, key_s, key_t = self.backbone(x) # [bs, 2048, 16, 8]
        if self.avgpool is not None:
            key_t = self.avgpool(key_t) # [bs, 2048, 1, 1]
            key_t = key_t.view(key_t.shape[0], -1) # [bs, 2048]
        return val, key_s, key_t
