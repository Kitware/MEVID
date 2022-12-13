import torch
import torch.nn as nn
from models.SRA import SRA
from models.TRA import TRA

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

class STAM(nn.Module):

    def __init__(self, inplanes, mid_planes, num, **kwargs):

        super(STAM, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.num = num

        self.Embeding = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu
        )
        self.Embeding.apply(weights_init_kaiming)

        self.TRAG = TRA(inplanes=inplanes, num=num)
        self.SRAG = SRA(inplanes=inplanes, num=num)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=inplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes),
            self.relu
        )
        self.conv_block.apply(weights_init_kaiming)

    def forward(self, feat_map):

        b, t, c, h, w = feat_map.size()
        reshape_map = feat_map.view(b * t, c, h, w)
        feat_vect = self.avg(reshape_map).view(b, t, -1)
        embed_feat = self.Embeding(reshape_map).view(b, t, -1, h, w)

        gap_feat_map0 = self.TRAG(feat_map, reshape_map, feat_vect, embed_feat)
        gap_feat_map = self.SRAG(feat_map, reshape_map, embed_feat, feat_vect, gap_feat_map0)
        gap_feat_map = self.conv_block(gap_feat_map)
        gap_feat_map = gap_feat_map.view(b, -1, c, h, w)
        torch.cuda.empty_cache()

        return gap_feat_map





