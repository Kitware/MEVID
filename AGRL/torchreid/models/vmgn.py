from __future__ import absolute_import
from __future__ import division

__all__ = ['vmgn']

import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

from torchreid.utils.reidtools import calc_splits
from torchreid.utils.torchtools import weights_init_kaiming, weights_init_classifier

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class GraphLayer(nn.Module):
    """
    graph block with residual learning.
    """

    def __init__(self, in_features, out_features, learn_graph=True, use_pose=True,
                 dist_method='l2', gamma=0.1, k=4, **kwargs):
        """
        :param in_features: input feature size.
        :param out_features: output feature size.
        :param learn_graph: learn a affinity graph or not.
        :param use_pose: use graph from pose estimation or not.
        :param dist_method: calculate the similarity between the vertex.
        :param k: nearest neighbor size.
        :param kwargs:
        """
        super(GraphLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learn_graph = learn_graph
        self.use_pose = use_pose
        self.dist_method = dist_method
        self.gamma = gamma

        assert use_pose or learn_graph
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.LeakyReLU(0.1)

        if self.learn_graph and dist_method == 'dot':
            num_hid = self.in_features // 8
            self.emb_q = nn.Linear(out_features, num_hid)
            self.emb_k = nn.Linear(out_features, num_hid)

        self._init_params()

    def get_sim_matrix(self, v_feats):
        """
        generate similarity matrix
        :param v_feats: (batch, num_vertex, num_hid)
        :return: sim_matrix: (batch, num_vertex, num_vertex)
        """
        if self.dist_method == 'dot':
            emb_q = self.emb_q(v_feats)
            emb_k = self.emb_k(v_feats)
            sim_matrix = torch.bmm(emb_q, emb_k.transpose(1, 2))
        elif self.dist_method == 'l2':
            # calculate the pairwise distance with exp(x) - 1 / exp(x) + 1
            distmat = torch.pow(v_feats, 2).sum(dim=2).unsqueeze(1) + \
                      torch.pow(v_feats, 2).sum(dim=2).unsqueeze(2)
            distmat -= 2 * torch.bmm(v_feats, v_feats.transpose(1, 2))
            distmat = distmat.clamp(1e-12).sqrt()  # numerical stability
            sim_matrix = 2 / (distmat.exp() + 1)
        else:
            raise NotImplementedError
        return sim_matrix

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

    def forward(self, input, adj):
        """
        :param input: (b, num_vertex, num_hid), where num_vertex = num_scale * seq_len * num_splits
        :param adj: (b, num_vertex, num_vertex), the pose-driven graph
        :return:
        """
        h = self.linear(input)
        N, V, C = h.size()

        # mask = torch.ones((N, V, V)).to(h.device)
        # for i in range(mask.size(1)):
        #     mask[:, i, i] = 0

        if self.use_pose:
            # adj = mask * adj
            adj = F.normalize(adj, p=1, dim=2)

        if self.learn_graph:
            graph = self.get_sim_matrix(input)
            # graph = mask * graph
            graph = F.normalize(graph, p=1, dim=2)
            if self.use_pose:
                graph = (adj + graph) / 2
        else:
            graph = adj

        h_prime = torch.bmm(graph, h)
        h_prime = self.bn(h_prime.view(N * V, -1)).view(N, V, -1)
        h_prime = self.relu(h_prime)

        return (1 - self.gamma) * input + self.gamma * h_prime


class ResNetBackbone(nn.Module):
    """
    Residual network
    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, layers, last_stride=2):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

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


class GSTA(nn.Module):
    def __init__(self, num_classes, loss, block, layers,
                 num_split, pyramid_part, num_gb, use_pose, learn_graph,
                 consistent_loss, nonlinear='relu', **kwargs):
        self.inplanes = 64
        super(GSTA, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion

        # backbone network
        backbone = ResNetBackbone(block, layers, 1)
        init_pretrained_weights(backbone, model_urls['resnet50'])
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4_1 = backbone.layer4
        self.layer4_2 = copy.deepcopy(self.layer4_1)

        # global branch, from layer4_1
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.global_bottleneck.bias.requires_grad_(False)
        self.global_classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        weights_init_kaiming(self.global_bottleneck)
        weights_init_classifier(self.global_classifier)

        # attention branch, from layer4_2
        self.num_split = num_split
        self.total_split_list = calc_splits(num_split) if pyramid_part else [num_split]
        self.total_split = sum(self.total_split_list)

        self.parts_avgpool = nn.ModuleList()
        for n in self.total_split_list:
            self.parts_avgpool.append(nn.AdaptiveAvgPool2d((n, 1)))

        # graph layers
        self.num_gb = num_gb
        self.graph_layers = nn.ModuleList()
        for i in range(num_gb):
            self.graph_layers.append(GraphLayer(in_features=self.feature_dim,
                                                out_features=self.feature_dim,
                                                use_pose=use_pose,
                                                learn_graph=learn_graph))

        self.consistent_loss = consistent_loss

        self.att_bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.att_bottleneck.bias.requires_grad_(False)
        self.att_classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        weights_init_kaiming(self.att_bottleneck)
        weights_init_classifier(self.att_classifier)

    def _attention_op(self, feat):
        """
        do attention fusion
        :param feat: (batch, seq_len, num_split, c)
        :return: feat: (batch, num_split, c)
        """
        att = F.normalize(feat.norm(p=2, dim=3, keepdim=True), p=1, dim=1)
        f = feat.mul(att).sum(dim=1)
        return f

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x4_1 = self.layer4_1(x)
        x4_2 = self.layer4_2(x)
        return x4_1, x4_2

    def forward(self, x, adj, *args):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        x4_1, x4_2 = self.featuremaps(x)
        _, c, h, w = x4_1.shape

        # global branch
        x4_1 = x4_1.view(B, S, c, h, w).transpose(1, 2).contiguous()
        g_f = self.global_avg_pool(x4_1).view(B, -1)
        g_bn = self.global_bottleneck(g_f)

        # attention branch
        v_f = list()
        for idx, n in enumerate(self.total_split_list):
            v_f.append(self.parts_avgpool[idx](x4_2).view(B, S, c, n))
        v_f = torch.cat(v_f, dim=3)
        f = v_f.transpose(2, 3).contiguous().view(B, S * self.total_split, c)

        # graph propagation
        for i in range(self.num_gb):
            f = self.graph_layers[i](f, adj)
        f = f.view(B, S, self.total_split, c)

        f_fuse = self._attention_op(f)

        att_f = f_fuse.mean(dim=1).view(B, -1)
        att_bn = self.att_bottleneck(att_f)

        if not self.training:
            return torch.cat([g_bn, att_bn], dim=1)

        g_out = self.global_classifier(g_bn)
        att_out = self.att_classifier(att_bn)

        # consistent
        if self.consistent_loss and self.training:
            satt_f_list = list()
            satt_out_list = list()
            # random select sub frames
            assert S >= 5
            for num_frame in [S-3, S-2, S-1]:
                sub_index = torch.randperm(S)[:num_frame]
                sub_index = torch.sort(sub_index)[0]
                sub_index = sub_index.long().to(f.device)
                sf = torch.gather(f, dim=1, index=sub_index.view(1, num_frame, 1, 1).repeat(B, 1, self.total_split, c))
                sf_fuse = self._attention_op(sf)
                satt_f = sf_fuse.mean(dim=1).view(B, -1)
                satt_bn = self.att_bottleneck(satt_f)
                satt_out = self.att_classifier(satt_bn)
                satt_f_list.append(satt_f)
                satt_out_list.append(satt_out)

        if self.loss == {'xent'}:
            out_list = [g_out, att_out]
            if self.consistent_loss:
                out_list.extend(satt_out_list)
            return out_list
        elif self.loss == {'xent', 'htri'}:
            out_list = [g_out, att_out]
            f_list = [g_f, att_f]
            if self.consistent_loss:
                out_list.extend(satt_out_list)
                f_list.extend(satt_f_list)
            return out_list, f_list
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
    print("Initialized model with pretrained weights from {}".format(model_url))


def vmgn(num_classes, loss, last_stride, num_split, num_gb, num_scale,
         pyramid_part, use_pose, learn_graph, consistent_loss=False, **kwargs):
    model = GSTA(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=last_stride,
        num_split=num_split,
        pyramid_part=pyramid_part,
        num_gb=num_gb,
        use_pose=use_pose,
        learn_graph=learn_graph,
        consistent_loss=consistent_loss,
        nonlinear='relu',
        **kwargs
    )

    return model