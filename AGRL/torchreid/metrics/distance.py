from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

import torch
from torch.nn import functional as F


def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
CurTime: 2019-08-03 00:12:21	Epoch: [160][100/117]	Time 0.663 (0.713)	Speed 89.744 samples/s	Data 0.0002 (0.0366)	Xent 1.0156 (1.0432)	Htri 0.0000 (0.0000)	Top1 1.0000 (0.9998)	Eta 10:12:04
Sat Aug  3 00:12:27 2019 ==> Test
Sat Aug  3 00:12:36 2019 Extracted features for query set, obtained 1980-by-1024 matrix
Sat Aug  3 00:13:08 2019 Extracted features for gallery set, obtained 9330-by-1024 matrix
Sat Aug  3 00:13:08 2019 ==> BatchTime(s)/BatchSize(img): 0.045/256
Sat Aug  3 00:13:08 2019 Computing distance matrix with metric=euclidean ...
Sat Aug  3 00:13:08 2019 Computing CMC and mAP
Sat Aug  3 00:13:12 2019 Results ----------
Sat Aug  3 00:13:12 2019 mAP: 78.00%
        torch.Tensor: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> input1 = torch.rand(10, 2048)
       >>> input2 = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(input1, input2)
       >>> distmat.size() # (10, 100)
    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input1.dim())
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input2.dim())
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )
    
    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    distmat = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat