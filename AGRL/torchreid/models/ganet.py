from __future__ import absolute_import
from __future__ import division

__all__ = ['ganet']

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

from torchreid.utils.reidtools import calc_splits

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


class PAM_Module(nn.Module):
    """
    Position attention module
    """
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        attention_mask = out.view(m_batchsize, C, height, width)

        out = self.gamma * attention_mask + x
        return out, attention_mask


class CAM_Module(nn.Module):
    """
    Channel attention module
    """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.channel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        max_energy_0 = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)
        energy_new = max_energy_0 - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class GraphLayer(nn.Module):
    """
    graph block with residual learning.
    """

    def __init__(self, in_features, out_features, learn_graph=True, use_pose=True,
                 dist_method='l2', gamma=0, k=4, **kwargs):
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

        mask = torch.ones((N, V, V)).to(h.device)
        for i in range(mask.size(1)):
            mask[:, i, i] = 0

        if self.use_pose:
            adj = mask * adj
            adj = F.normalize(adj, p=1, dim=2)

        if self.learn_graph:
            graph = self.get_sim_matrix(input)
            graph = mask * graph
            graph = F.normalize(graph, p=1, dim=2)
            if self.use_pose:
                graph = (adj + graph) / 2
        else:
            graph = adj

        h_prime = torch.bmm(graph, h)
        h_prime = self.bn(h_prime.view(N * V, -1)).view(N, V, -1)
        h_prime = self.relu(h_prime)

        return input + self.gamma * h_prime


class GSTA(nn.Module):
    def __init__(self, num_classes, loss, block, layers,
                 num_split, pyramid_part, num_gb, use_pose, learn_graph,
                 consistent_loss, nonlinear='relu', **kwargs):
        self.inplanes = 64
        super(GSTA, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # attention
        self.num_split = num_split
        self.total_split_list = calc_splits(num_split) if pyramid_part else [num_split]
        self.total_split = sum(self.total_split_list)

        self.pam_layer = PAM_Module(self.feature_dim)
        self.cam_layer = CAM_Module(self.feature_dim)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # graph layers
        self.num_gb = num_gb
        self.graph_layers = nn.ModuleList()
        for i in range(num_gb):
            self.graph_layers.append(GraphLayer(in_features=self.feature_dim,
                                                out_features=self.feature_dim,
                                                use_pose=use_pose,
                                                learn_graph=learn_graph))

        self.consistent_loss = consistent_loss

        self.bottleneck = nn.BatchNorm1d((self.num_gb + 1) * self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear((self.num_gb + 1) * self.feature_dim, num_classes, bias=False)

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

    def _attention_op(self, feat):
        """
        do attention fusion
        :param feat: (batch, seq_len, num_split, c)
        :return: feat: (batch, num_split, c)
        """
        att = F.normalize(feat.norm(p=2, dim=3, keepdim=True), p=1, dim=1)
        f = feat.mul(att).sum(dim=1)
        return f

    def forward(self, x, adj, *args):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W)
        f = self.featuremaps(x)
        _, c, h, w = f.shape

        # spatial and channel attention
        pyra_f = list()
        for n in self.total_split_list:
            slice_step = h // n
            for i in range(n):
                slice_f = f[:, :, slice_step * i: slice_step * (i + 1)]
                pyra_f.append(slice_f)

        v_f = list()
        for i in range(self.total_split):
            pam_f, _ = self.pam_layer(pyra_f[i])
            # cam_f = self.cam_layer(pyra_f[i])
            tmp_f = pam_f + pyra_f[i]
            v_f.append(self.avgpool(tmp_f).view(B * S, c))
        v_f = torch.stack(v_f, dim=2)
        f = v_f.transpose(1, 2).contiguous()
        f = f.view(B, S * self.total_split, c)
        # graph propagation
        gl_out = [f]
        for i in range(self.num_gb):
            gl_out.append(self.graph_layers[i](gl_out[-1], adj))
        f = torch.cat(gl_out, dim=2).view(B, S, self.total_split, (self.num_gb + 1) * c)

        f_fuse = self._attention_op(f)

        f_g = f_fuse.mean(dim=1).view(B, -1)
        bn = self.bottleneck(f_g)

        # consistent
        if self.consistent_loss and self.training:
            # random select sub frames
            sub_index = list()
            for i in range(B):
                tmp_ind = list(range(0, S))
                tmp_ind.remove(np.random.randint(S))
                sub_index.append(tmp_ind)
            sub_index = torch.LongTensor(sub_index).to(f_fuse.device)
            sf = torch.gather(f, dim=1, index=sub_index.view(B, S - 1, 1, 1).repeat(1, 1, f.size(2), f.size(3)))
            sf_fuse = self._attention_op(sf)
            sf_g = sf_fuse.mean(dim=1).view(B, -1)
            sbn = self.bottleneck(sf_g)
            sy = self.classifier(sbn)

        if not self.training:
            return bn

        y = self.classifier(bn)

        if self.loss == {'xent'}:
            if self.consistent_loss:
                return [y, sy]
            else:
                return y
        elif self.loss == {'xent', 'htri'}:
            if self.consistent_loss:
                return [y, sy], [f_g, sf_g]
            else:
                return y, f_g
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


def ganet(num_classes, loss, last_stride, num_split, num_gb, num_scale, knn,
         pyramid_part, use_pose, learn_graph, pretrained=True, consistent_loss=False, **kwargs):
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
    if pretrained:
        print('init pretrained weights from {}'.format(model_urls['resnet50']))
        init_pretrained_weights(model, model_urls['resnet50'])
    return model