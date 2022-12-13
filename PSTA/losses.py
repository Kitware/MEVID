from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['CrossEntropyLabelSmooth', 'TripletLoss', 'CenterLoss']

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.softmax_list = nn.Softmax(dim=2)

    def forward(self, inputs, targets, frame_feat = None):

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

        if frame_feat is not None:
            selfpace_loss = self.self_pace(inputs, frame_feat)
            return loss + selfpace_loss
        else:
            return loss

    def self_pace(self, tracklet_feat, frame_feat, bata = 0.1):
        tracklet_probs = self.softmax(tracklet_feat)
        frame_probs = self.softmax_list(frame_feat)
        max_index = torch.max(torch.max(frame_probs, 2)[0], 1)[1]

        good_frame_vect = []
        for i in range(frame_probs.size(0)):
            good_frame_vect.append(frame_probs[i, max_index[i]])

        good_frame_vect = torch.stack(good_frame_vect, 0)
        loss = 1/2 * (-torch.log(tracklet_probs) * good_frame_vect - torch.log(good_frame_vect) * tracklet_probs)
        loss = bata * loss.mean(0).sum()
        return loss

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        #dist = []
        #for i in range(batch_size):
        #    value = distmat[i][mask[i]]
        #    value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
        #    dist.append(value)
        #dist = torch.cat(dist)
        #loss = dist.mean()
        return loss

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, distance='cosine'):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
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
        elif self.distance == 'cosine':
            fnorm = torch.norm(inputs, p=2, dim=1, keepdim=True)
            l2norm = inputs.div(fnorm.expand_as(inputs))
            dist = - torch.mm(l2norm, l2norm.t())

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

def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        if cfg.DISTANCE == "cosine":
            triplet = CosineTripletLoss(cfg.SOLVER.MARGIN)
        elif cfg.DISTANCE == "euclid":
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if cfg.SOLVER.CENTER_ON:
                    return xent(score, target) + \
                            triplet(feat, target)[0] \
                            + cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
                else:
                    return xent(score, target) + \
                           triplet(feat, target)[0]
            else:
                return F.cross_entropy(score, target) + \
                        triplet(feat, target)[0] + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center'
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion



class CosineTripletLoss(nn.Module):

    def __init__(self, margin=0.3):
        super(CosineTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        inputs = inputs / (inputs ** 2).sum(dim=1, keepdim=True).sqrt()
        dist = 1 - inputs.mm(inputs.t())
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().reshape(1))
            dist_an.append(dist[i][mask[i] == 0].min().reshape(1))
        dist_ap = torch.cat(dist_ap, -1)
        dist_an = torch.cat(dist_an, -1)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss, dist_an, dist_ap


class TripletLossAttrWeightes(nn.Module):

    def __init__(self, margin=0, dis_type="cosine"):
        super(TripletLossAttrWeightes, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.dis_type = dis_type

    def forward(self, inputs, targets, weights_vector):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # w_dis = euclidean_dist(weights_vector, weights_vector)
        # w_dis.argmax()
        # Compute pairwise distance, replace by the official when merged
        if self.dis_type == "cosine":
            inputs = inputs / (inputs ** 2).sum(dim=1, keepdim=True).sqrt()
            dist = 1 - inputs.mm(inputs.t())
        elif self.dis_type == "euclid":
            dist = euclidean_dist(inputs, inputs)
        w_dis = dist
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]][w_dis[i][mask[i]].argmax()].reshape(1))
            dist_an.append(dist[i][mask[i]][w_dis[i][mask[i]].argmin()].reshape(1))
        dist_ap = torch.cat(dist_ap, -1)
        dist_an = torch.cat(dist_an, -1)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
# if __name__ == '__main__':
#     input = torch.randn((16,651))
#     targets = torch.randperm(651)
#     loss = CrossEntropyLabelSmooth(num_classes=651)
#     loss(input,targets)