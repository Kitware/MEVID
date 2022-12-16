from __future__ import absolute_import

import torch
import math
import copy
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.nn import functional as F

from models import inflate
from .resnets1 import resnet50_s1
from .DAO import DAO
from .TKS import TKS


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        # init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Bottleneck3d(nn.Module):

    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        self.conv1 = inflate.inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)
        self.conv2 = inflate.inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate.inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = self._inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None

    def _inflate_downsample(self, downsample2d, time_stride=1):
        downsample3d = nn.Sequential(
            inflate.inflate_conv(downsample2d[0], time_dim=1, 
                                 time_stride=time_stride),
            inflate.inflate_batch_norm(downsample2d[1]))
        return downsample3d

    def forward_once(self, x):
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

    def forward(self, x):
        """xh: [bs, c, t, h, w]
           xl: [bs, c, t, h//2, w//2]
        """
        assert type(x) is tuple
        xh, xl = x
        
        outh = self.forward_once(xh)
        outl = self.forward_once(xl)

        return outh, outl


class BiCnet_TKS(nn.Module):

    def __init__(self, num_classes, **kwargs):

        super(BiCnet_TKS, self).__init__()
        resnet2d = resnet50_s1(pretrained=True)

        self.conv1 = inflate.inflate_conv(resnet2d.conv1, time_dim=1)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(resnet2d.maxpool, time_dim=1)

        self.downsample = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.layer1 = self._inflate_reslayer(resnet2d.layer1)
        self.layer1_h2l = nn.Conv3d(256, 256*3, 1, stride=1, padding=0, bias=False)

        self.layer2 = self._inflate_reslayer(resnet2d.layer2)
        self.layer2_h2l = nn.Conv3d(512, 512*3, 1, stride=1, padding=0, bias=False)
        self.TKS = TKS(in_channel=512)

        self.layer3 = self._inflate_reslayer(resnet2d.layer3)
        self.layer3_h2l = nn.Conv3d(1024, 1024*3, 1, stride=1, padding=0, bias=False)
        self.DAO = DAO(in_channel=1024)

        self.layer4 = self._inflate_reslayer(resnet2d.layer4)

        # fc using random initialization
        add_block = nn.BatchNorm1d(2048)
        add_block.apply(weights_init_kaiming)
        self.bn = add_block
        
        # classifier using Random initialization
        classifier = nn.Linear(2048, num_classes)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def _inflate_reslayer(self, reslayer2d):
        reslayers3d = []
        for layer2d in reslayer2d:
            layer3d = Bottleneck3d(layer2d)
            reslayers3d.append(layer3d)

        return nn.Sequential(*reslayers3d)

    def pooling(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b*t, c, h, w)
        x = F.max_pool2d(x, x.size()[2:])
        x = x.view(b, t, -1)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        b, c, t, h, w = x.size()

        xh = torch.cat((x[:, :, 0:1], x[:, :, 4:5]), 2) #[b, c, 2, h, w]
        xl = torch.cat((x[:, :, 1:4], x[:, :, 5:]), 2) #[b, c, K, h, w]
        xl = self.downsample(xl) #[b, c, K, h//2, w//2]

        # layer1
        xh, xl = self.layer1((xh, xl))
        x_h2l = self.layer1_h2l(self.downsample(xh)) #[b, 256*K, 2, 32, 16]
        x_h2l = x_h2l.view(b, 3, 256, *x_h2l.size()[2:]) #[b, K, 256, 2, 32, 16]
        x_h2l = x_h2l.permute(0, 2, 3, 1, 4, 5) #[b, 256, 2, K, 32, 16]
        x_h2l = x_h2l.contiguous().view(b, 256, -1, 32, 16)
        xl = xl + x_h2l

        # layer2
        xh, xl = self.layer2((xh, xl))
        x_h2l = self.layer2_h2l(self.downsample(xh)) 
        x_h2l = x_h2l.view(b, 3, 512, *x_h2l.size()[2:]) 
        x_h2l = x_h2l.permute(0, 2, 3, 1, 4, 5) 
        x_h2l = x_h2l.contiguous().view(b, 512, -1, 16, 8)
        xl = xl + x_h2l

        xh, xl = self.TKS(xh, xl)

        # layer3
        xh, xl = self.layer3((xh, xl))
        x_h2l = self.layer3_h2l(self.downsample(xh)) 
        x_h2l = x_h2l.view(b, 3, 1024, *x_h2l.size()[2:]) 
        x_h2l = x_h2l.permute(0, 2, 3, 1, 4, 5) 
        x_h2l = x_h2l.contiguous().view(b, 1024, -1, 8, 4)
        xl = xl + x_h2l

        xh, xl, masks = self.DAO(xh, xl)

        # layer4
        xh, xl = self.layer4((xh, xl))

        xh = self.pooling(xh) #[bs, 2, c]
        xl = self.pooling(xl) #[bs, K, c]

        xh = xh.mean(1, keepdims=True) #[b, 1, c]
        xl = xl.mean(1, keepdims=True) #[b, 1, c]

        x = 0.5 * xh + 0.5 * xl #[b, 1, c]

        if not self.training:
            return x

        x = x.mean(1)
        f = self.bn(x)
        y = self.classifier(f)

        return y, f, masks
