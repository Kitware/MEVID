import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from torchreid.utils.torchtools import weights_init_xavier
from torchreid.utils.reidtools import calc_splits


def pose_aggregation(features, locs, num_scale=1, total_parts=[3], seq_len=4):
    """
    :param features: (b, num_scale * total_split, seq_len, c)
    :param locs: (b, seq_len, num_parts)
    :param num_scale: int
    :param total_parts: list(int)
    :param seq_len: int
    :return:
    """
    locs_t = locs.transpose(1, 2)
    num_parts = total_parts[-1]
    total_split = sum(total_parts)
    batch, _, _, channel = features.size()
    if total_split in [3, 4]:
        locs_tmp = locs_t * seq_len + (torch.arange(seq_len).unsqueeze(0).repeat(num_parts, 1)).float().unsqueeze(0).to(
            locs_t.device)
        features = features.view(batch, num_scale, num_parts * seq_len, channel)
        fused_features = torch.zeros((batch, num_scale, num_parts, channel)).to(features.device)
        for b in range(batch):
            loc = locs_tmp[b].data.cpu().numpy().reshape(num_parts * seq_len).astype(int)
            for idx_p in range(num_parts):
                index = loc[idx_p * seq_len: (idx_p + 1) * seq_len]
                fused = []
                for idx in index:
                    fused.append(features[b, :, idx])
                fused_features[b, :, idx_p] = torch.stack(fused, dim=1).mean(dim=1)
    elif total_split in [6, 7]:
        head_num_parts = 3
        features = features.view(batch, num_scale, total_split, seq_len, channel)
        fused_features_tail = torch.zeros((batch, num_scale, num_parts, channel)).to(features.device)
        fused_features_head = features[:, :, :head_num_parts].mean(dim=3)
        features_tail = features[:, :, head_num_parts:].view(batch, num_scale, num_parts * seq_len, channel)
        locs_tmp = locs_t * seq_len + (torch.arange(seq_len).unsqueeze(0).repeat(num_parts, 1)).float().unsqueeze(0).to(
            locs_t.device)
        for b in range(batch):
            loc = locs_tmp[b].data.cpu().numpy().reshape(num_parts * seq_len).astype(int)
            for idx_p in range(num_parts):
                index = loc[idx_p * seq_len: (idx_p + 1) * seq_len]
                fused = []
                for idx in index:
                    fused.append(features_tail[b, :, idx])
                fused_features_tail[b, :, idx_p] = torch.stack(fused, dim=1).mean(dim=1)
        fused_features = torch.cat([fused_features_head, fused_features_tail], dim=2)
    else:
        raise NotImplementedError
    return fused_features.view(batch, num_scale * total_split, channel)


class GraphBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0., alpha=1, gamma=1,
                 learn_graph=True, use_pose=True, self_loop=False, **kwargs):
        super(GraphBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.gamma = gamma
        self.learn_graph = learn_graph
        self.use_pose = use_pose
        self.self_loop = self_loop

        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.normal_(self.linear.weight, mean=0, std=0.001)

        if self.learn_graph:
            num_hid = 128
            self.emb_q = nn.Linear(out_features, num_hid)
            self.emb_k = nn.Linear(out_features, num_hid)
            nn.init.normal_(self.emb_q.weight, std=0.001)
            nn.init.constant_(self.emb_q.bias, 0)
            nn.init.normal_(self.emb_k.weight, std=0.001)
            nn.init.constant_(self.emb_k.bias, 0)

        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, input, adj):
        h = self.linear(input)
        N, V, C = h.size()

        if self.use_pose:
            adj = F.normalize(adj, p=1, dim=2)

        if self.learn_graph:
            emb_q = self.emb_q(h)
            emb_k = self.emb_k(h)
            graph = torch.bmm(emb_q, emb_k.transpose(1, 2))
            graph = F.softmax(graph, dim=2)
            if self.self_loop:
                I = torch.eye(V, device=input.device).view(1, V, V).repeat(N, 1, 1)
                graph = F.softmax(graph + I, dim=2)
            if self.use_pose:
                graph = (adj + self.gamma * graph) / (1 + self.gamma)
        else:
            graph = adj

        h_prime = torch.bmm(graph, h)
        h_prime = F.dropout(h_prime, p=self.dropout, training=self.training)
        h_prime = F.relu(h_prime)

        h_prime = h_prime.view(N * V, self.out_features)
        h_prime = self.bn(h_prime)
        h_prime = h_prime.view(N, V, self.out_features)

        assert input.size() == h_prime.size(), 'when use skip connection, input size must equal to output size.'
        return input + self.alpha * h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MSPyraPartNet(nn.Module):
    """use layer2 layer3 layer4."""
    def __init__(self, num_classes=100, loss={'xent', 'htri'}, num_split=4,
                 **kwargs):
        super(MSPyraPartNet, self).__init__()
        self.num_classes = num_classes
        self.loss = loss
        self.num_parts = num_split

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4

        self.num_scale = 3
        self.total_parts = calc_splits(self.num_parts)
        self.total_split = sum(self.total_parts)
        self.num_hid = 512

        self.avg_pool = nn.ModuleList()
        self.max_pool = nn.ModuleList()
        for n in self.total_parts:
            self.avg_pool.append(nn.AdaptiveAvgPool2d((n, 1)))
            self.max_pool.append(nn.AdaptiveMaxPool2d((n, 1)))

        self.reduce_f1 = nn.Linear(512, self.num_hid)
        self.bn_f1 = nn.BatchNorm1d(self.num_hid)
        self.reduce_f2 = nn.Linear(1024, self.num_hid)
        self.bn_f2 = nn.BatchNorm1d(self.num_hid)
        self.reduce_f3 = nn.Linear(2048, self.num_hid)
        self.bn_f3 = nn.BatchNorm1d(self.num_hid)
        weights_init_xavier(self.reduce_f1)
        weights_init_xavier(self.reduce_f2)
        weights_init_xavier(self.reduce_f3)
        weights_init_xavier(self.bn_f1)
        weights_init_xavier(self.bn_f2)
        weights_init_xavier(self.bn_f3)

        self.fusion_conv = nn.Conv1d(self.num_scale * self.total_split, 1, 1, bias=False)
        self.classifier = nn.ModuleList()
        for i in range(self.num_scale * self.total_split + 1):
            self.classifier.append(nn.Linear(self.num_hid, num_classes))
        weights_init_xavier(self.fusion_conv)
        weights_init_xavier(self.classifier)

    def forward(self, x, adj=None):
        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f1 = self.layer2(x)
        f2 = self.layer3(f1)
        f3 = self.layer4(f2)

        # global and local feature
        l_f1 = []
        l_f2 = []
        l_f3 = []
        for idx, n in enumerate(self.total_parts):
            l_f1.append((self.avg_pool[idx](f1) + self.max_pool[idx](f1)).view(b, s, 512, -1))
            l_f2.append((self.avg_pool[idx](f2) + self.max_pool[idx](f2)).view(b, s, 1024, -1))
            l_f3.append((self.avg_pool[idx](f3) + self.max_pool[idx](f3)).view(b, s, 2048, -1))
        l_f1 = torch.cat(l_f1, dim=3).permute(0, 3, 1, 2).contiguous()
        l_f1 = self.bn_f1(self.reduce_f1(l_f1).view(b * self.total_split * s, -1)).view(b, self.total_split * s, -1)
        l_f2 = torch.cat(l_f2, dim=3).permute(0, 3, 1, 2).contiguous()
        l_f2 = self.bn_f2(self.reduce_f2(l_f2).view(b * self.total_split * s, -1)).view(b, self.total_split * s, -1)
        l_f3 = torch.cat(l_f3, dim=3).permute(0, 3, 1, 2).contiguous()
        l_f3 = self.bn_f3(self.reduce_f3(l_f3).view(b * self.total_split * s, -1)).view(b, self.total_split * s, -1)
        f = torch.cat([l_f1, l_f2, l_f3], dim=1).view(b, self.num_scale * self.total_split, s, -1)

        vf = f.mean(dim=2)
        allf = [vf[:, i] for i in range(self.num_scale * self.total_split)]

        fused_f = self.fusion_conv(vf).view(b, self.num_hid)
        if not self.training:
            return fused_f

        allf.append(fused_f)
        y = [self.classifier[i](vf[:, i]) for i in range(self.num_scale * self.total_split)]
        y.append(self.classifier[-1](fused_f))
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, allf
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class MSPyraPartGraphNet(nn.Module):
    def __init__(self, num_classes=100, loss={'xent', 'htri'}, num_split=3, use_pose=True,
                 learn_graph=True, num_gb=3, **kwargs):
        super(MSPyraPartGraphNet, self).__init__()
        self.num_classes = num_classes
        self.loss = loss
        resnet50 = torchvision.models.resnet50(pretrained=True)
        # 0: conv2d 1: bn 2: relu 3: maxpool 4:layer1 5: layer2 6:layer3 7: layer4
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.num_scale = 3  # number of layers for feature extraction
        self.num_split = num_split  # number of feature split in spatial pooling
        self.total_split = sum(calc_splits(num_split))
        self.num_hid = 512
        self.num_gb = num_gb
        self.use_pose = use_pose

        self.avg_pool = nn.ModuleList()
        self.max_pool = nn.ModuleList()
        for n in calc_splits(self.num_split):
            self.avg_pool.append(nn.AdaptiveAvgPool2d((n, 1)))
            self.max_pool.append(nn.AdaptiveMaxPool2d((n, 1)))

        self.reduce1 = nn.Linear(512, self.num_hid)
        self.bn1 = nn.BatchNorm1d(self.num_hid)
        self.reduce2 = nn.Linear(1024, self.num_hid)
        self.bn2 = nn.BatchNorm1d(self.num_hid)
        self.reduce3 = nn.Linear(2048, self.num_hid)
        self.bn3 = nn.BatchNorm1d(self.num_hid)
        weights_init_xavier(self.reduce1)
        weights_init_xavier(self.reduce2)
        weights_init_xavier(self.reduce3)
        weights_init_xavier(self.bn1)
        weights_init_xavier(self.bn2)
        weights_init_xavier(self.bn3)

        self.gbs = nn.ModuleList()
        for j in range(self.num_gb):
            self.gbs.append(
                GraphBlock(in_features=self.num_hid, out_features=self.num_hid,
                           learn_graph=learn_graph, use_pose=use_pose))

        self.fusion_conv = nn.Conv1d(self.num_scale * self.total_split, 1, 1, bias=False)
        self.classifiers = nn.ModuleList()
        for i in range(self.num_scale * self.total_split + 1):
            self.classifiers.append(nn.Linear(self.num_hid * (self.num_gb + 1), num_classes))
        weights_init_xavier(self.fusion_conv)
        weights_init_xavier(self.classifiers)

    def forward(self, x, adj):
        b, s, c, h, w = x.size()
        x = x.view(b * s, c, h, w)
        for name, module in self.base._modules.items():
            if name == '3':  # maxpool
                x = module(x)
            elif name == '5':  # layer2
                layer2 = module(x)
            elif name == '6':  # layer3
                layer3 = module(layer2)
            elif name == '7':  # layer4
                layer4 = module(layer3)
            else:
                x = module(x)
        # global and local feature
        l2_f = []
        l3_f = []
        l4_f = []
        for idx, n in enumerate(calc_splits(self.num_split)):
            l2_f.append((self.avg_pool[idx](layer2) + self.max_pool[idx](layer2)).view(b, s, 512, -1))
            l3_f.append((self.avg_pool[idx](layer3) + self.max_pool[idx](layer3)).view(b, s, 1024, -1))
            l4_f.append((self.avg_pool[idx](layer4) + self.max_pool[idx](layer4)).view(b, s, 2048, -1))
        l2_f = torch.cat(l2_f, dim=3).permute(0, 3, 1, 2).contiguous()
        l3_f = torch.cat(l3_f, dim=3).permute(0, 3, 1, 2).contiguous()
        l4_f = torch.cat(l4_f, dim=3).permute(0, 3, 1, 2).contiguous()

        l2_f = self.reduce1(l2_f).view(b * self.total_split * s, self.num_hid)
        l2_f = self.bn1(l2_f).view(b, self.total_split * s, self.num_hid)
        l3_f = self.reduce2(l3_f).view(b * self.total_split * s, self.num_hid)
        l3_f = self.bn2(l3_f).view(b, self.total_split * s, self.num_hid)
        l4_f = self.reduce3(l4_f).view(b * self.total_split * s, self.num_hid)
        l4_f = self.bn3(l4_f).view(b, self.total_split * s, self.num_hid)
        f = torch.cat([l2_f, l3_f, l4_f], dim=1)

        gb_out = [f]
        for i in range(self.num_gb):
            gb_out.append(self.gbs[i](gb_out[-1], adj))
        f = torch.stack(gb_out, dim=2).view(b, self.num_scale * self.total_split, s, (self.num_gb + 1) * self.num_hid)

        vf = f.mean(dim=2)
        allf = [vf[:, i] for i in range(self.num_scale * self.total_split)]

        fused_f = self.fusion_conv(vf).view(b, (self.num_gb + 1) * self.num_hid)
        allf.append(fused_f)

        if not self.training:
            return fused_f
        y = [self.classifiers[i](vf[:, i]) for i in range(self.num_scale * self.total_split)]
        y.append(self.classifiers[-1](fused_f))

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, allf
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
