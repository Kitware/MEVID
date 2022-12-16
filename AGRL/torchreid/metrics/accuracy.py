from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        def calc_acc(output, target):
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                acc = correct_k.mul_(1.0 / batch_size)
                res.append(acc.item())
            return res

        all_res = []
        if isinstance(output, (tuple, list)):
            for out in output:
                all_res.append(calc_acc(out, target))
        else:
            all_res.append(calc_acc(output, target))
        return np.array(all_res)