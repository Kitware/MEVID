import math
from bisect import bisect_right

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CyclicLR(_LRScheduler):

    def __init__(self, optimizer, base_lr, max_lr, step_size, gamma=0.99, mode='triangular', last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        assert mode in ['triangular', 'triangular2', 'exp_range']
        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        new_lr = []
        # make sure that the length of base_lrs doesn't change. Dont care about the actual value
        for base_lr in self.base_lrs:
            cycle = np.floor(1 + self.last_epoch / (2 * self.step_size))
            x = np.abs(float(self.last_epoch) / self.step_size - 2 * cycle + 1)
            if self.mode == 'triangular':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
            elif self.mode == 'triangular2':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) / float(2 ** (cycle - 1))
            elif self.mode == 'exp_range':
                lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * (self.gamma ** (
                    self.last_epoch))
            new_lr.append(lr)
        return new_lr

class CyclicCosAnnealingLR(_LRScheduler):
    r"""

    Implements reset on milestones inspired from CosineAnnealingLR pytorch

    Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))
    When last_epoch > last set milestone, lr is automatically set to \eta_{min}
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        decay_milestones(list of ints):List of increasing epoch indices. Ideally,decay values should overlap with milestone points
        gamma (float): factor by which to decay the max learning rate at each decay milestone
        eta_min (float): Minimum learning rate. Default: 1e-6
        last_epoch (int): The index of last epoch. Default: -1.


    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, milestones, decay_milestones=None, gamma=0.5, eta_min=1e-6, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
            self.eta_min = eta_min
            self.milestones = milestones
            self.milestones2 = decay_milestones

            self.gamma = gamma
            super(CyclicCosAnnealingLR, self).__init__(optimizer, last_epoch)


    def get_lr(self):
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx - 1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        if self.milestones2:
            return [
                self.eta_min + (base_lr * self.gamma ** bisect_right(self.milestones2, self.last_epoch) - self.eta_min) *
                (1 + math.cos(math.pi * curr_pos / width)) / 2
                for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * curr_pos / width)) / 2
                    for base_lr in self.base_lrs]


class CyclicLinearLR(_LRScheduler):
    r"""
    Implements reset on milestones inspired from Linear learning rate decay

    Set the learning rate of each parameter group using a linear decay
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart:
    .. math::
        \eta_t = \eta_{min} + (\eta_{max} - \eta_{min})(1 -\frac{T_{cur}}{T_{max}})
    When last_epoch > last set milestone, lr is automatically set to \eta_{min}

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list of ints): List of epoch indices. Must be increasing.
        decay_milestones(list of ints):List of increasing epoch indices. Ideally,decay values should overlap with milestone points
        gamma (float): factor by which to decay the max learning rate at each decay milestone
        eta_min (float): Minimum learning rate. Default: 1e-6
        last_epoch (int): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, milestones, decay_milestones=None, gamma=0.5, eta_min=1e-6, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.eta_min = eta_min

        self.gamma = gamma
        self.milestones = milestones
        self.milestones2 = decay_milestones
        super(CyclicLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]

        idx = bisect_right(self.milestones, self.last_epoch)

        left_barrier = 0 if idx == 0 else self.milestones[idx - 1]
        right_barrier = self.milestones[idx]

        width = right_barrier - left_barrier
        curr_pos = self.last_epoch - left_barrier

        if self.milestones2:
            return [self.eta_min + (
                        base_lr * self.gamma ** bisect_right(self.milestones2, self.last_epoch) - self.eta_min) *
                    (1. - 1.0 * curr_pos / width)
                    for base_lr in self.base_lrs]

        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1. - 1.0 * curr_pos / width)
                    for base_lr in self.base_lrs]

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            self.after_scheduler.step(metrics, epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                return self.after_scheduler.step(epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]