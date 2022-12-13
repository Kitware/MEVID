import torch
import torch.nn as nn
from torch.nn import functional as F

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


class SRA(nn.Module):

    def __init__(self, inplanes, num):

        super(SRA, self).__init__()

        self.inplanes = inplanes
        self.num = num

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        print('Build ' + self.num + ' layer SRA!')

        self.alphi_appearance = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes/8),
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes/8)),
            self.relu
        )
        self.alphi_appearance.apply(weights_init_kaiming)

        self.delta_appearance = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8),
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes / 8)),
            self.relu
        )
        self.delta_appearance.apply(weights_init_kaiming)

        self.gg_spatial = nn.Sequential(
            nn.Conv2d(in_channels=2 * 16 * 8, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
        )
        self.gg_spatial.apply(weights_init_kaiming)

        self.spa_para = nn.Sequential(
            nn.Conv2d(in_channels=2 * 128, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
            nn.Conv2d(in_channels=128, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            self.sigmoid
        )
        self.spa_para.apply(weights_init_kaiming)

        self.app_channel = nn.Sequential(
            nn.Linear(in_features=inplanes, out_features=int(inplanes / 8)),
            self.relu,
            nn.Linear(in_features=int(inplanes / 8), out_features=inplanes),
            self.sigmoid
        )
        self.app_channel.apply(weights_init_kaiming)

    def forward(self, feat_map, re_featmap, Embeding_feature, feat_vect, aggregative_feature=None):

        b, t, c, h, w = feat_map.size()
        Embeding_feat = Embeding_feature.view(b*t, -1,  h, w)
        alphi_feat = self.alphi_appearance(re_featmap).view(b * t, -1, h * w)
        delta_feat = self.delta_appearance(re_featmap).view(b * t, -1, h * w)
        alphi_feat = alphi_feat.permute(0, 2, 1)
        Gs = torch.matmul(alphi_feat, delta_feat)
        Gs_in = Gs.permute(0, 2, 1).view(b * t, h * w, h, w)
        Gs_out = Gs.view(b * t, h * w, h, w)
        Gs_joint = torch.cat((Gs_in, Gs_out), 1)
        Gs_joint = self.gg_spatial(Gs_joint)
        para_spa = torch.cat((Embeding_feat, Gs_joint), 1)
        para_spa = self.spa_para(para_spa).view(b, t, -1, h, w)

        aggregative_feature_list = []
        for i in range(0, t, 2):

            para_0 = self.app_channel(feat_vect[:, i, :]).view(b, -1, 1, 1)
            para_1 = self.app_channel(feat_vect[:, i + 1, :]).view(b, -1, 1, 1)

            para_0 = para_0 * para_spa[:, i, :, :, :]
            para_1 = para_1 * para_spa[:, i + 1, :, :, :]

            aggregative_feature_list.append(aggregative_feature[:, int(i/2), :, :, :] + self.relu(para_0 * feat_map[:, i, :, :, :] + para_1 * feat_map[:, i + 1, :, :, :]))

        aggregative_features = torch.stack(aggregative_feature_list, 1)
        aggregative_features = aggregative_features.view(b * aggregative_features.size(1), -1, h, w)
        torch.cuda.empty_cache()

        return aggregative_features