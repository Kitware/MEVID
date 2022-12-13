import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

def init_pretrained_weight(model,model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url, model_dir = './')
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k,v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
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
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias,0.0)


class app_tem(nn.Module):

    def __init__(self, num_classes,  last_stride, model_path, model_name, pretrain_choice, global_refine_method, local_refine_method, seq_len):
        super(app_tem, self).__init__()
        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet()

        if pretrain_choice == 'imagenet':
            init_pretrained_weight(self.base,model_urls[model_name])
            print('Loading pretrained ImageNet model......')

        self.seq_len = seq_len
        self.global_refine_method = global_refine_method
        self.local_refine_method = local_refine_method
        self.num_classes = num_classes

        self.part_num = 4
        self.feat_pool = nn.AdaptiveAvgPool2d((1,1))
        self.local_part_avgpool = nn.AdaptiveAvgPool2d((self.part_num,1))
        self.pool_3D = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.plances = 512

        self.relu = nn.ReLU(inplace = True)

        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels = self.in_planes, out_channels = self.plances,
                                              kernel_size = 1) for _ in range(seq_len - 1)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(self.plances) for _ in range(seq_len - 1)])

        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels = self.plances, out_channels = self.plances,
                                              kernel_size = 3, padding = 1, stride = 1) for _ in range(seq_len - 1)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(self.plances) for _ in range(seq_len - 1)])

        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels = self.plances, out_channels = self.in_planes,
                                              kernel_size = 1) for _ in range(seq_len - 1)])
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(self.in_planes) for _ in range(seq_len - 1)])

        self.channel_conv = nn.ModuleList([nn.Conv2d(in_channels = self.in_planes * 3, out_channels = self.in_planes,
                                                     kernel_size = 1) for _ in range(seq_len - 1)])

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)

        self.conv1.apply(weights_init_kaiming)
        self.conv2.apply(weights_init_kaiming)
        self.conv3.apply(weights_init_kaiming)
        self.channel_conv.apply(weights_init_kaiming)

        self.bn1.apply(weights_init_kaiming)
        self.bn2.apply(weights_init_kaiming)
        self.bn3.apply(weights_init_kaiming)

        # self.sum_bottleneck.apply(weights_init_kaiming)
        # self.sum_classifier.apply(weight_init_classifier)

    def residual_block(self, feat_map, num):

        b, t, c, w, h = feat_map.size()
        gap_feat_map = []

        for i in range(t - 1):
            gap_map = feat_map[:, i, :, :, :] - feat_map[:, i+1, :, :, :]
            gap_feat_map.append(gap_map)

        gap_feat_map = torch.stack(gap_feat_map, 1)
        gap_feat_map = gap_feat_map ** 2
        gap_feat_map = gap_feat_map.view(b*(t - 1), c, w, h)

        gap_feat_map = self.conv1[num](gap_feat_map)
        gap_feat_map = self.bn1[num](gap_feat_map)

        gap_feat_map = self.conv2[num](gap_feat_map)
        gap_feat_map = self.bn2[num](gap_feat_map)

        gap_feat_map = self.conv3[num](gap_feat_map)
        gap_feat_map = self.bn3[num](gap_feat_map)

        gap_feat_map = self.relu(gap_feat_map)
        gap_feat_map = gap_feat_map.view(b, t - 1, c, w, h)
        return gap_feat_map

    def forward(self, x, pids = None, camids = None):

        b, t, c, w, h = x.size()
        x = x.view(b * t, c, w, h)
        feat_map = self.base(x)                                                                  # (b*t,c,16,8)
        w = feat_map.size(2)
        h = feat_map.size(3)
        feat_map = feat_map.view(b, t, self.in_planes, w, h)       # (32, 4, 2048, 16, 8)
        # dense_feat_list = []

        for i in range(self.seq_len -1):
            gap_feat_map = self.residual_block(feat_map, i)
            dense_feat_map = []

            for i in range(gap_feat_map.size(1)):
                dense_map = torch.cat([gap_feat_map[:, i, :, :, :], feat_map[:, i, :, :, :], feat_map[:, i + 1, :, :, :]], 1)
                dense_feat_map.append(dense_map)

            dense_feat_map = torch.stack(dense_feat_map, 1)
            dense_feat_map = dense_feat_map.view(-1, self.in_planes * 3, w, h)
            dense_feat_map = self.channel_conv[i](dense_feat_map)
            feat_map = dense_feat_map.view(b, -1, self.in_planes, w, h)

        feat_map = feat_map.view(b, self.in_planes, w, h)
        feature = self.feat_pool(feat_map).view(b, self.in_planes)
        BN_feature = self.bottleneck(feature)



        if self.training:
            cls_score = self.classifier(BN_feature)
            return cls_score, feature
        else:
            return BN_feature, pids, camids

if __name__ == '__main__':
    x = torch.rand(16,5,3,256,128)
    num_classes = 625
    last_stride = 1
    model_path = '/home/wyq/.torch/models/resnet50-19c8e357.pth'
    neck = 'no'
    model_name = 'resnet50'
    pretrain_choice = 'imagenet'
    model = app_tem(num_classes,last_stride,model_path,neck,model_name,pretrain_choice)
    model(x)