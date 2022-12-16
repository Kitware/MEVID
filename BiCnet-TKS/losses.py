from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
"""

class DivRegLoss(nn.Module):
    def __init__(self, detach=True, sqrt=True):
        super(DivRegLoss, self).__init__()
        print('detach: {}'.format(detach))
        self.detach = detach
        self.sqrt = sqrt

    def forward_once(self, p1, p2):
        """p1: [bs, k], p2: [bs, k]
        """
        bs, k = p1.size()

        I = torch.eye(2, dtype=p1.dtype).cuda()
        x = torch.stack((p1, p2), 1) #[bs, 2, k]
        if self.sqrt:
            x = torch.sqrt(x)
        tmp = torch.bmm(x, x.transpose(1, 2)) #[bs, 2, 2]
        tmp = tmp - I.unsqueeze(0)
        tmp = tmp.view(bs, -1)
        tmp = torch.norm(tmp, dim=1) / tmp.size(1)
        loss = tmp.mean()
        return loss

    def forward(self, inputs):
        """inputs: [[bs, k], [bs, k], [bs, k]]
        """
        p1, p2, p3 = inputs
        if self.detach:
            p1 = p1.detach()
        loss1 = self.forward_once(p1, p2)
        loss2 = self.forward_once(p1, p3)
        loss3 = self.forward_once(p2, p3)
        return (loss1 + loss2 + loss3) / 3
		

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.5, distance='consine', use_gpu=True):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'consine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.use_gpu = use_gpu
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'euclidean':
            dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, inputs, inputs.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        elif self.distance == 'consine':
            fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
            l2norm = inputs.div(fnorm.expand_as(inputs))
            dist = - torch.mm(l2norm, l2norm.t())

        if self.use_gpu: targets = targets.cuda()
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, scale=16, **kwargs):
        print('contrastive loss with scale {}'.format(scale))
        super(ContrastiveLoss, self).__init__()
        self.scale = scale

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, p=2, dim=1)
        similarities = torch.matmul(inputs, inputs.t()) * self.scale

        targets = targets.view(-1,1)
        mask = torch.eq(targets, targets.T).float().cuda()
        mask_self = torch.eye(targets.size(0)).float().cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        # compute log_prob
        exp_logits = torch.exp(similarities) * (1 - mask_self)
        # log_prob = similarities - torch.log(exp_logits.sum(1, keepdim=True))
        log_sum_exp_pos_and_all_neg = torch.log((exp_logits * mask_neg).sum(1, keepdim=True) + exp_logits)
        log_prob = similarities - log_sum_exp_pos_and_all_neg

        # compute mean of log-likelihood over positive
        loss = (mask_pos * log_prob).sum(1) / mask_pos.sum(1)

        loss = - loss.mean()

        return loss
