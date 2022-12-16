from __future__ import absolute_import

import random
import math
import numpy as np


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = list(frame_indices)

        while len(out) < self.size:
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)

        return out


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size=4):
        self.size = size

    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)
        size = self.size

        if len(frame_indices) >= (size - 1) * 8 + 1:
            out = frame_indices[0: (size - 1) * 8 + 1: 8]
        elif len(frame_indices) >= (size - 1) * 4 + 1:
            out = frame_indices[0: (size - 1) * 4 + 1: 4]
        elif len(frame_indices) >= (size - 1) * 2 + 1:
            out = frame_indices[0: (size - 1) * 2 + 1: 2]
        elif len(frame_indices) >= size:
            out = frame_indices[0:size:1]
        else:
            out = frame_indices[0:size]
            while len(out) < size:
                for index in out:
                    if len(out) >= size:
                        break
                    out.append(index)

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size=4, stride=8):
        self.size = size
        self.stride = stride

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)

        if len(frame_indices) >= self.size * self.stride:
            rand_end = len(frame_indices) - (self.size - 1) * self.stride - 1
            begin_index = random.randint(0, rand_end)
            end_index = begin_index + (self.size - 1) * self.stride + 1
            out = frame_indices[begin_index:end_index:self.stride]
        elif len(frame_indices) >= self.size:
            index = np.random.choice(len(frame_indices), size=self.size, replace=False)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        else:
            index = np.random.choice(len(frame_indices), size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]

        return out


class TemporalRestrictedCrop(object):
    """Temporally divide the video into N chunks of equation duration.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)
        seq_len = len(frame_indices)

        if seq_len < self.size:
            index = np.random.choice(seq_len, size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        else:
            out = []
            x0 = 0
            for k in range(self.size):
                x1 = x0 + (seq_len - k - 1) / self.size + 1
                chuck_frame_indices = frame_indices[x0: x1]
                index = np.random.choice(len(chuck_frame_indices), size=1, replace=False)
                out.append(chuck_frame_indices[index[0]])
                x0 = x1
        return out


class TemporalRestrictedBeginCrop(object):
    """Temporally divide the video into N chunks of equation duration.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)
        seq_len = len(frame_indices)

        if seq_len < self.size:
            index = np.random.choice(seq_len, size=self.size, replace=True)
            index.sort()
            out = [frame_indices[index[i]] for i in range(self.size)]
        else:
            out = []
            x0 = 0
            for k in range(self.size):
                x1 = x0 + (seq_len - k - 1) / self.size + 1
                chuck_frame_indices = frame_indices[x0: x1]
                out.append(chuck_frame_indices[0])
                x0 = x1
        return out


class TemporalRestrictedTest(object):
    """Temporally divide the video into N chunks of equation duration.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        frame_indices = list(frame_indices)
        seq_len = len(frame_indices)

        if seq_len < self.size:
            return frame_indices
        else:
            out = []
            x0 = 0
            for k in range(self.size):
                x1 = x0 + (seq_len - k - 1) / self.size + 1
                chuck_frame_indices = frame_indices[x0: x1]
                out.append(chuck_frame_indices)
                x0 = x1

            cnt = [0] * self.size
            out1 = []
            while len(out1) < seq_len:
                for k in range(self.size):
                    if cnt[k] < len(out[k]):
                        out1.append(out[k][cnt[k]])
                        cnt[k] += 1
        return out1


if __name__ == '__main__':
    net = TemporalRestrictedTest(size=6)
    x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'q', 'i', 's', 't', 'u', '1', '2', '3', '4', '5', 'x', 'y', 'z']
    print(len(x))
    y = net(x)
    print(y)
