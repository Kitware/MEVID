import torch
import torch.nn as nn
import torch.nn.functional as F

import model
import model.resnet as res
import model.memory as mem

import parser
args = parser.parse_args()

class STMN(nn.Module):
    def __init__(self, feat_dim=2048, num_class=710, stride=1):
        super(STMN, self).__init__()
        self.features = res.Resnet50(stride=stride)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.smem = mem.SpatialMemory(feat_dim=feat_dim, mem_size=args.smem_size, margin=args.smem_margin)
        self.tmem = mem.TemporalMemory(feat_dim=feat_dim, mem_size=args.tmem_size,
                                       margin=args.tmem_margin, seq_len=args.seq_len)

        self.bn_s = nn.BatchNorm1d(feat_dim)
        self.bn_s.bias.requires_grad_(False)
        self.bn_s.apply(model.weights_init_kaiming)
        self.bn_t = nn.BatchNorm1d(feat_dim)
        self.bn_t.bias.requires_grad_(False)
        self.bn_t.apply(model.weights_init_kaiming)

        self.cls_s = nn.Linear(feat_dim, num_class,  bias=False)
        self.cls_s.apply(model.weights_init_classifier)
        self.cls_t = nn.Linear(feat_dim, num_class,  bias=False)
        self.cls_t.apply(model.weights_init_classifier)

    def forward(self, x):
        B, S, C, H, W = x.size()
        val, key_s, key_t = self.features(x.reshape((B*S, )+x.shape[2:]))

        # spatial
        smem = self.smem(key_s, val)
        val_s = smem['out']
        val_s_bn = self.bn_s(val_s)

        # temporal
        tmem = self.tmem(key_t, val_s)
        val_t = tmem['out']
        val_t_bn = self.bn_t(val_t)

        if self.training:
            val_s_cls = self.cls_s(val_s_bn)
            val_t_cls = self.cls_t(val_t_bn)
            return {'val_s':val_s, 'val_s_cls':val_s_cls, 'smem':smem,
                    'val_t':val_t, 'val_t_cls':val_t_cls, 'tmem':tmem}
        else:
            return {'val_bn':val_t_bn, 'smem':smem, 'tmem':tmem, 'val_t': val_t}
