# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
import torch.nn.functional as F
import numpy as np
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class AttributeRecogModule(nn.Module):
    def __init__(self, in_planes, num_class):
        super(AttributeRecogModule, self).__init__()
        self.in_planes = in_planes
        self.attention_conv = nn.Conv2d(in_planes, 1, 1)
        weights_init_kaiming(self.attention_conv)
        self.classifier = nn.Linear(in_planes, num_class)
        self.classifier.apply(weights_init_classifier)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        a = self.attention_conv(x)
        a = torch.sigmoid(a)
        a = a.view(b, t, 1, x.size(2), x.size(3))

        atten = a.expand(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = atten * x.view(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = global_feat.view(b * t, self.in_planes, global_feat.size(3), global_feat.size(4))
        global_feat = self.gap(global_feat)
        # global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat = global_feat.view(b, t, -1)
        global_feat = F.relu(torch.mean(global_feat, 1))
        y = self.classifier(global_feat)
        return y,a

class MultiAttributeRecogModule(nn.Module):
    def __init__(self, in_planes, num_classes=[]):
        super(MultiAttributeRecogModule, self).__init__()
        self.in_planes = in_planes
        self.out_planes = in_planes // 2
        self.conv = nn.Conv2d(self.in_planes, self.out_planes, 1)
        self.bn = nn.BatchNorm2d(self.out_planes)
        self.attr_recog_modules = nn.ModuleList([AttributeRecogModule(self.out_planes, n) for n in num_classes])
    def forward(self, x, b, t):
        ys = []
        attens = []
        local_feature = self.conv(x)
        local_feature = F.relu(self.bn(local_feature))
        local_feature = local_feature.view(b, t, self.out_planes, local_feature.size(2), local_feature.size(3))
        for m in self.attr_recog_modules:
            y, a = m(local_feature)
            ys.append(y)
            attens.append(a)
        return ys, torch.cat(attens, 2)

class MultiAttributeRecogModuleBCE(nn.Module):
    def __init__(self, in_planes, num_classes=[]):
        super(MultiAttributeRecogModuleBCE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = in_planes
        self.conv = nn.Conv2d(self.in_planes, self.out_planes, 1)
        weights_init_kaiming(self.conv)
        self.bn = nn.BatchNorm2d(self.out_planes)
        weights_init_kaiming(self.bn)
        self.attention_conv = nn.Conv2d(self.out_planes, 1, 3, padding=1)
        weights_init_kaiming(self.attention_conv)
        self.classifier = nn.Linear(in_planes, sum(num_classes))
        self.classifier.apply(weights_init_classifier)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, b, t):
        local_feature = self.conv(x)
        local_feature = F.relu(self.bn(local_feature))
        a = self.attention_conv(local_feature)
        a = torch.sigmoid(a)
        a = a.view(b, t, 1, x.size(2), x.size(3))
        # a = torch.nn.functional.softmax(a, 1)

        atten = a.expand(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = atten * x.view(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = global_feat.view(b * t, self.in_planes, global_feat.size(3), global_feat.size(4))
        global_feat = self.gap(global_feat)
        # global_feat = global_feat.view(global_feat.shape[0], -1)
        global_feat = global_feat.view(b, t, -1)
        global_feat = F.relu(torch.mean(global_feat, 1))
        if self.training:
            y = self.classifier(global_feat)
        else:
            y = None
        return y, atten

class VideoBaseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(VideoBaseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.attention_conv = nn.Conv2d(self.in_planes, 1, 3, padding=1)
        weights_init_kaiming(self.attention_conv)
        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b * t, x.size(2), x.size(3), x.size(4))
        global_feat = self.base(x) # (b, 2048, 1, 1)

        a = self.attention_conv(global_feat)
        a = a.view(b, t, 1, global_feat.size(2), global_feat.size(3))
        a = torch.sigmoid(a)
        a = a.expand(b, t, self.in_planes, global_feat.size(2), global_feat.size(3))

        global_feat = a * global_feat.view(b, t, self.in_planes, global_feat.size(2), global_feat.size(3))
        global_feat = global_feat.view(b * t, self.in_planes, global_feat.size(3), global_feat.size(4))
        global_feat= self.gap(global_feat)
        global_feat = global_feat.view(b, t, -1)
        global_feat = global_feat.permute(0, 2, 1)
        global_feat = torch.mean(global_feat, 2, keepdim=True)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
        feat = feat.squeeze(2)
        global_feat = global_feat.squeeze(2)
        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
