from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
from itertools import chain

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

def compute_pids_and_pids_dict(data_source):

    index_dic = defaultdict(list)
    for index, (_, pid, _, _, _) in enumerate(data_source):
        index_dic[pid].append(index)
    pids = list(index_dic.keys())
    return pids, index_dic


class ReIDBatchSampler(Sampler):

    def __init__(self, data_source, p: int, k: int):

        self._p = p
        self._k = k

        pids, index_dic = compute_pids_and_pids_dict(data_source)

        self._unique_labels = np.array(pids)
        self._label_to_items = index_dic.copy()

        self._num_iterations = len(self._unique_labels) // self._p

    def __iter__(self):

        def sample(set, n):
            if len(set) < n:
                return np.random.choice(set, n, replace=True)
            return np.random.choice(set, n, replace=False)

        np.random.shuffle(self._unique_labels)

        for k, v in self._label_to_items.items():
            random.shuffle(self._label_to_items[k])

        curr_p = 0

        for idx in range(self._num_iterations):
            p_labels = self._unique_labels[curr_p: curr_p + self._p]
            curr_p += self._p
            batch = [sample(self._label_to_items[l], self._k) for l in p_labels]
            batch = list(chain(*batch))
            yield batch

    def __len__(self):
        return self._num_iterations
