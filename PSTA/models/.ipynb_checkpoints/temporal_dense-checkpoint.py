import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *
from models.backbone.densenet import *


class gap_block(nn.Module):

    def __init__(self, inplanes, mid_planes, seq_len):
        super(gap_block, self).__init__()
        
#         d = torch.tensor(inplanes).float()
#         self.cat_para = nn.ParameterList([nn.Parameter(torch.randn(inplanes, 1, 1) / torch.sqrt(2 / d)) for i in range(2)])
        self.conv1 = nn.Conv2d(in_channels = inplanes, out_channels = mid_planes, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.conv2 = nn.Conv2d(in_channels = mid_planes, out_channels = mid_planes, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = nn.Conv2d(in_channels = mid_planes, out_channels = inplanes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(inplanes)
        
        self.cat_conv = nn.Conv1d(in_channels = 3, out_channels = 1, kernel_size = 3, padding = 1)
        self.para_fc_0 = nn.Linear(in_features = inplanes, out_features = int(inplanes / 16))
        self.para_fc_1 = nn.Linear(in_features = int(inplanes / 16), out_features = inplanes)
        
        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        
        if seq_len != 1:
            self.cat_conv2 = nn.Conv1d(in_channels = seq_len, out_channels = 1, kernel_size = 3, padding = 1)
            self.cat_conv2.apply(weights_init_kaiming)
        
        self.cat_conv.apply(weights_init_kaiming)
        self.para_fc_0.apply(weights_init_kaiming)
        self.para_fc_1.apply(weights_init_kaiming)
        
        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        self.bn3.apply(weights_init_kaiming)

        self.conv1.apply(weights_init_kaiming)
        self.conv2.apply(weights_init_kaiming)
        self.conv3.apply(weights_init_kaiming)

    def forward(self, feat_map):
        b, t, c, w, h = feat_map.size()
        feat_para = self.avg(feat_map.view(b*t, c, w, h)).view(b*t, c)
        para = self.sigmoid(self.para_fc_0(feat_para))
        para = self.relu(self.para_fc_1(para))
        para = para.view(b, t, -1, 1, 1)
        gap_feat_map = []

        for idx in range(0, t, 2):
            gap_map = para[:, idx + 1, :, :, :] * feat_map[:, idx, :, :, :] + para[:, idx, :, :, :] * feat_map[:, idx + 1, :, :, :]
            gap_map = gap_map ** 2
            gap_feat_map.append(gap_map)

        gap_feat_map = torch.stack(gap_feat_map, 1)
        seq_len = gap_feat_map.size(1)
        gap_feat_map = gap_feat_map.view(b * seq_len, -1, w, h)

        gap_feat_map = self.conv1(gap_feat_map)
        gap_feat_map = self.bn1(gap_feat_map)
        gap_feat_map = self.relu(gap_feat_map)
        
        gap_feat_map = self.conv2(gap_feat_map)
        gap_feat_map = self.bn2(gap_feat_map)
        gap_feat_map = self.relu(gap_feat_map)

        gap_feat_map = self.conv3(gap_feat_map)
        gap_feat_map = self.bn3(gap_feat_map)
        gap_feat_map = self.relu(gap_feat_map)
        
        
        gap_feat_vect = self.avg(gap_feat_map).view(b, seq_len, -1)
        feat_vect = self.avg(feat_map.view(b*t, c, w, h)).view(b, t, -1)
        
        feature_list = []
        for i in range(seq_len):
            idx = 2 * i 
            stack_feature = torch.stack([gap_feat_vect[:, i, :], feat_vect[:, idx, :], feat_vect[:, idx + 1, :]], 1)
            feature = self.cat_conv(stack_feature)
            feature_list.append(feature)
            
        feat = torch.cat(feature_list, 1)
        
        if seq_len != 1:
            feature = self.cat_conv2(feat)
            feature = self.relu(feature)
        
        feature = feature.view(b, -1)
        gap_feat_map = gap_feat_map.view(b, seq_len, -1, w, h)
        
        return gap_feat_map, feature


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def init_pretrained_weight(model, model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url, model_dir='./')
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


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


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class tem_dense(nn.Module):

    def __init__(self, num_classes, model_name, pretrain_choice, seq_len):
        super(tem_dense, self).__init__()
        self.in_planes = 2048
        self.base = ResNet()

        if pretrain_choice == 'imagenet':
            init_pretrained_weight(self.base, model_urls[model_name])
            print('Loading pretrained ImageNet model......')

        self.seq_len = seq_len
        self.num_classes = num_classes

        self.feat_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.plances = 1024
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances, kernel_size=1)
        self.cat_conv0 = nn.Conv1d(in_channels = seq_len, out_channels = 1, kernel_size = 3, padding = 1)
        
        t = int(seq_len / 2)
        self.layer1 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t)
        t = int(t / 2)
        self.layer2 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t)
        t = int(t /2)
        self.layer3 = gap_block(inplanes=self.plances, mid_planes=int(self.plances * 0.5), seq_len = t)

        self.cat_conv1 = nn.Conv1d(in_channels=4, out_channels=1, kernel_size = 3, padding = 1)
#                 self.cat_bn1 = nn.BatchNorm2d(self.plances)

        self.bottleneck = nn.BatchNorm1d(self.plances)
        self.classifier = nn.Linear(self.plances, self.num_classes, bias=False)
        self.bottleneck.bias.requires_grad_(False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)

    def forward(self, x, pids=None, camids=None):

        b, t, c, w, h = x.size()
        x = x.view(b * t, c, w, h)
        feat_map = self.base(x)  # (b * t, c, 16, 8)
        w = feat_map.size(2)
        h = feat_map.size(3)
        feat_map = self.conv1(feat_map)
        feat_map = self.relu(feat_map)

        feat_map_0 = feat_map.view(b, t, -1, w, h)  # (32, seq_len, 1048, 16, 8)
        feat_map_1, feat_vect_1 = self.layer1(feat_map_0)
        feat_map_2, feat_vect_2 = self.layer2(feat_map_1)
        feat_map_3, feat_vect_3 = self.layer3(feat_map_2)

        feat_vect_0 = self.cat_conv0(self.feat_pool(feat_map.view(b*t, -1, w, h)).view(b, t, -1))
        feat_vect_0 = feat_vect_0.view(b, -1)
        feat_vect_0 = self.relu(feat_vect_0)
#         feat_vect_1 = self.pool3d(feat_map_1.permute(0, 2, 1, 3, 4)).view(b, -1)
#         feat_vect_2 = self.pool3d(feat_map_2.permute(0, 2, 1, 3, 4)).view(b, -1)
#         feat_vect_3 = self.pool3d(feat_map_3.permute(0, 2, 1, 3, 4)).view(b, -1)

        cat_feat = torch.stack([feat_vect_0, feat_vect_1, feat_vect_2, feat_vect_3], 1)
        feature = self.cat_conv1(cat_feat).view(b, -1)

        feature = feature.view(b, self.plances)
        BN_feature = self.bottleneck(feature)
        cls_score_list = []
        torch.cuda.empty_cache()

        if self.training:
            cls_score_list.append(self.classifier(BN_feature))
            return cls_score_list, feature
        else:
            return BN_feature, pids, camids

