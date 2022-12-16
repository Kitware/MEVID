from __future__ import absolute_import
from __future__ import division

__all__ = ['sta_p4']

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class STA(nn.Module):
    """Part-based Convolutional Baseline.

    Reference:
        STA: Spatial-Temporal Attention for Large-Scale Video-based Person Re-Identification

    Public keys:
        - ``sta``.
    """

    def __init__(self, num_classes, loss, block, layers,
                 reduced_dim=512,
                 nonlinear='relu',
                 enable_reg=False,
                 **kwargs):
        self.inplanes = 64
        super(STA, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.parts = 4

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # sta layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Sequential(
            nn.Linear(4096, reduced_dim, bias=False),
            nn.BatchNorm1d(reduced_dim),
            nn.ReLU()
        )

        self.feature_dim = reduced_dim
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self._init_params()

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

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x, *args):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        f = self.featuremaps(x)
        _, c, h, w = f.shape

        # attention map, first l2 normalization
        g_a = f.norm(p=2, dim=1, keepdim=True).view(B * S, 1, h * w)
        g_a = F.normalize(g_a, p=2, dim=2).view(B * S, 1, h, w)

        # spatial attention map, second l1 norm
        s_a = self.parts_avgpool(g_a).view(B, S, self.parts)

        # temporal attention map, third l1 norm
        t_a = F.normalize(s_a, p=1, dim=1)

        v_g = self.parts_avgpool(f).view(B, S, c, self.parts)

        # highest score index
        h_index = t_a.argmax(dim=1, keepdim=True).unsqueeze(2)

        # f_1
        f_1 = v_g.gather(dim=1, index=h_index.expand((B, 1, c, self.parts))).view(B, c, self.parts)

        # f_2
        f_2 = v_g.mul(t_a.unsqueeze(2)).sum(dim=1)

        # fusion
        f_fuse = torch.cat([f_1, f_2], dim=1)
        # f_fuse = f_2

        # GAP
        f_g = F.adaptive_avg_pool1d(f_fuse, 1).view(B, -1)

        f_t = self.fc1(f_g)

        if not self.training:
            return f_t

        y = self.classifier(f_t)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            # v_g = F.normalize(v_g, p=2, dim=1)
            return y, f_t
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def sta_p4(num_classes, loss={'xent', 'htri'}, last_stride=1, pretrained=True, **kwargs):
    model = STA(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=last_stride,
        reduced_dim=1024,
        nonlinear='relu',
        **kwargs
    )
    if pretrained:
        print('init pretrained weights from {}'.format(model_urls['resnet50']))
        init_pretrained_weights(model, model_urls['resnet50'])
    return model