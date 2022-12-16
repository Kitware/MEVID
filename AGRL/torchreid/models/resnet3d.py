import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet3d', 'resnet3d10', 'resnet3d18', 'resnet3d34', 'resnet3d50', 'resnet3d101',
    'resnet3d152', 'resnet3d200'
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class ResNet3d(nn.Module):
    def __init__(self, block, layers, shortcut_type='B', num_classes=400):
        self.inplanes = 64
        super(ResNet3d, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_matched_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            # if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
            print("loading " + name)
            own_state[name].copy_(param)

    def forward(self, x):
        # default size is (b, s, c, w, h), s for seq_len, c for channel
        # convert for 3d cnn, (b, c, s, w, h)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        x = x.view(x.size(0), -1)
        y = self.fc(x)

        return y


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet3d10(pretrained='', **kwargs):
    """Constructs a ResNet3d-10 model.
    """
    model = ResNet3d(BasicBlock, [1, 1, 1, 1], **kwargs)
    if pretrained:
        model = load_state_dict(model, pretrained)
    return model


def resnet3d18(pretrained='', **kwargs):
    """Constructs a ResNet3d-18 model.
    """
    model = ResNet3d(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model = load_state_dict(model, pretrained)
    return model


def resnet3d34(pretrained='', **kwargs):
    """Constructs a ResNet3d-34 model.
    """
    model = ResNet3d(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model, pretrained)
    return model


def resnet3d50(pretrained='', **kwargs):
    """Constructs a ResNet3d-50 model.
    """
    model = ResNet3d(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model, pretrained)
    return model


def resnet3d101(pretrained='', **kwargs):
    """Constructs a ResNet3d-101 model.
    """
    model = ResNet3d(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model, pretrained)
    return model


def resnet3d152(pretrained='', **kwargs):
    """Constructs a ResNet3d-101 model.
    """
    model = ResNet3d(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model, pretrained)
    return model


def resnet3d200(pretrained='', **kwargs):
    """Constructs a ResNet3d-101 model.
    """
    model = ResNet3d(Bottleneck, [3, 24, 36, 3], **kwargs)
    if pretrained:
        model = load_state_dict(model, pretrained)
    return model

def load_state_dict(model, pretrained):
    assert os.path.exists(pretrained), '{} is not exists'.format(os.path.abspath(pretrained))
    checkpoint = torch.load(pretrained, map_location='cpu')
    pretrain_dict = checkpoint['state_dict']
    state_dict = {}
    for key in pretrain_dict:
        state_dict[key.partition("module.")[2]] = pretrain_dict[key]
    pretrain_dict = state_dict

    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    assert len(pretrain_dict) == len(model_dict)
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    return model
