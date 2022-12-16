from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import h5py
import os.path as osp
from scipy.io import loadmat
import numpy as np


"""Dataset classes"""


class LSVID(object):
    """
    LS-VID

    Reference:
    Li J, Wang J, Tian Q, Gao W and Zhang S Global-Local Temporal Representations for Video Person Re-Identification[J]. ICCV, 2019

    Dataset statistics:
    # identities: 1261
    # tracklets: 8298 (train) + 1980 (query) + 11310 (gallery)
    # cameras: 15

    Note:
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.


    """

    def __init__(self, root=None, **kwargs):
        if root is None:
            root = '/mnt/local0/houruibing/data/re_id_data/video/LS-VID/'
        self.root = root
        self.train_name_path = osp.join(self.root, 'list_sequence/list_seq_train.txt')
        self.test_name_path = osp.join(self.root, 'list_sequence/list_seq_test.txt')
        self.test_query_IDX_path = osp.join(self.root, 'test/data/info_test.mat')

        self._check_before_run()

        # prepare meta data
        tracklet_train = self._get_names(self.train_name_path)
        tracklet_test = self._get_names(self.test_name_path)

        test_query_IDX = h5py.File(self.test_query_IDX_path, mode='r')['query'][0,:]
        test_query_IDX = np.array(test_query_IDX, dtype=int)

        test_query_IDX -= 1  # index from 0

        tracklet_test_query = tracklet_test[test_query_IDX, :]

        test_gallery_IDX = [i for i in range(tracklet_test.shape[0]) if i not in test_query_IDX]

        tracklet_test_gallery = tracklet_test[test_gallery_IDX, :]
        
        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(tracklet_train, home_dir='tracklet_train', relabel=True)
        train_dense, num_train_dense_tracklets, _, _ = \
            self._process_data(tracklet_train, home_dir='tracklet_train', relabel=True, sampling_step=64)

        test_query, num_test_query_tracklets, num_test_query_pids, num_test_query_imgs = \
            self._process_data(tracklet_test_query, home_dir='tracklet_test', relabel=False)
        test_gallery, num_test_gallery_tracklets, num_test_gallery_pids, num_test_gallery_imgs = \
            self._process_data(tracklet_test_gallery, home_dir='tracklet_test', relabel=False)

        num_imgs_per_tracklet = num_train_imgs + num_test_gallery_imgs + num_test_query_imgs #+ num_val_query_imgs + num_val_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_test_gallery_pids # + num_val_gallery_pids 
        num_total_tracklets = num_train_tracklets + num_test_gallery_tracklets + num_test_query_tracklets #+ num_val_query_tracklets + num_val_gallery_tracklets

        print("=> LS-VID loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset       | # ids | # tracklets")
        print("  ------------------------------")
        print("  train        | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  train_dense  | {:5d} | {:8d}".format(num_train_pids, num_train_dense_tracklets))
        print("  test_query   | {:5d} | {:8d}".format(num_test_query_pids, num_test_query_tracklets))
        print("  test_gallery | {:5d} | {:8d}".format(num_test_gallery_pids, num_test_gallery_tracklets))
        print("  ------------------------------")
        print("  total        | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.train_dense = train_dense
        self.query = test_query
        self.gallery = test_gallery

        self.num_train_pids = num_train_pids
        self.num_test_query_pids = num_test_query_pids
        self.num_test_gallery_pids = num_test_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.test_query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.test_query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                basepath, pid = new_line.split(' ')
                names.append([basepath, int(pid)])
        return np.array(names)

    def _process_data(self, meta_data, home_dir=None, relabel=False, sampling_step=0):
        assert home_dir in ['tracklet_train', 'tracklet_val', 'tracklet_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 1].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {int(pid): label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            tracklet_path = osp.join(self.root, meta_data[tracklet_idx, 0]) + '*'
            img_paths = glob.glob(tracklet_path)  # avoid .DS_Store
            img_paths.sort()
            pid = int(meta_data[tracklet_idx, 1])
            _, _, camid, _ = osp.basename(img_paths[0]).split('_')
            camid = int(camid)

            if relabel:
                pid = pid2label[pid]
            camid -= 1  # index starts from 0

            num_imgs_per_tracklet.append(len(img_paths))

            # dense sampling
            if sampling_step != 0:
                num_sampling = len(img_paths)//sampling_step
                if num_sampling == 0:
                    tracklets.append((img_paths, pid, camid))
                else:
                    for idx in range(num_sampling):
                        if idx == num_sampling - 1:
                            tracklets.append((img_paths[idx*sampling_step:], pid, camid))
                        else:
                            tracklets.append((img_paths[idx*sampling_step : (idx+1)*sampling_step], pid, camid))
            else:
                tracklets.append((img_paths, pid, camid))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


if __name__ == '__main__':
    LSVID()
    
