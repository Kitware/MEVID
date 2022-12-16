from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import json
from .bases import BaseImageDataset


"""Dataset classes"""

class iLIDSVID(BaseImageDataset):
    """
    iLIDS-VID

    Reference:
    Wang et al. Person Re-Identification by Video Ranking. ECCV 2014.
    
    Dataset statistics:
    # identities: 300
    # tracklets: 600
    # cameras: 2

    Args:
        split_id (int): indicates which split to use. There are totally 10 splits.
    """
    root = '/data/i-LIDS-VID'
    dataset_url = 'http://www.eecs.qmul.ac.uk/~xiatian/iLIDS-VID/iLIDS-VID.tar'

    def __init__(self, root='/data/i-LIDS-VID', split_id=0):
        self.root = root
        self.data_dir = osp.join(self.root, 'i-LIDS-VID')
        self.split_dir = osp.join(self.data_dir, 'train-test people splits')
        self.split_mat_path = osp.join(self.split_dir, 'train_test_splits_ilidsvid.mat')
        self.split_path = osp.join(self.data_dir, 'splits.json')
        self.cam_1_path = osp.join(self.data_dir, 'i-LIDS-VID/sequences/cam1')
        self.cam_2_path = osp.join(self.data_dir, 'i-LIDS-VID/sequences/cam2')

        if split_id == 0:
            self._download_data()
            self._check_before_run()

            self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        if split_id == 0:
            print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if split_id == 0:
            print("=> iLIDS-VID loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _download_data(self):
        if osp.exists(self.root):
            print("This dataset has been downloaded.")
            return

        mkdir_if_missing(self.root)
        fpath = osp.join(self.root, osp.basename(self.dataset_url))

        print("Downloading iLIDS-VID dataset")
        url_opener = urllib.request.URLopener()
        url_opener.retrieve(self.dataset_url, fpath)

        print("Extracting files")
        tar = tarfile.open(fpath)
        tar.extractall(path=self.root)
        tar.close()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_dir):
            raise RuntimeError("'{}' is not available".format(self.split_dir))

    def _prepare_split(self):
        if not osp.exists(self.split_path):
            print("Creating splits")
            mat_split_data = loadmat(self.split_mat_path)['ls_set']
            
            num_splits = mat_split_data.shape[0]
            num_total_ids = mat_split_data.shape[1]
            assert num_splits == 10
            assert num_total_ids == 300
            num_ids_each = int(num_total_ids/2)

            # pids in mat_split_data are indices, so we need to transform them
            # to real pids
            person_cam1_dirs = os.listdir(self.cam_1_path)
            person_cam2_dirs = os.listdir(self.cam_2_path)

            # make sure persons in one camera view can be found in the other camera view
            assert set(person_cam1_dirs) == set(person_cam2_dirs)

            splits = []
            for i_split in range(num_splits):
                # first 50% for testing and the remaining for training, following Wang et al. ECCV'14.
                train_idxs = sorted(list(mat_split_data[i_split,num_ids_each:]))
                test_idxs = sorted(list(mat_split_data[i_split,:num_ids_each]))
                
                train_idxs = [int(i)-1 for i in train_idxs]
                test_idxs = [int(i)-1 for i in test_idxs]
                
                # transform pids to person dir names
                train_dirs = [person_cam1_dirs[i] for i in train_idxs]
                test_dirs = [person_cam1_dirs[i] for i in test_idxs]
                
                split = {'train': train_dirs, 'test': test_dirs}
                splits.append(split)

            print("Totally {} splits are created, following Wang et al. ECCV'14".format(len(splits)))
            print("Split file is saved to {}".format(self.split_path))
            write_json(splits, self.split_path)

        print("Splits created")

    def _process_data(self, dirnames, cam1=True, cam2=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}

        #txt_name = 'ilids' + str(cam1) + str(cam2) + '.txt'
        #fid = open(txt_name, "w")    

        for dirname in dirnames:
            if cam1:
                person_dir = osp.join(self.cam_1_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0, 1))
                num_imgs_per_tracklet.append(len(img_names))
                #fid.write("cam1_" + dirname + '\n')
            if cam2:
                person_dir = osp.join(self.cam_2_path, dirname)
                img_names = glob.glob(osp.join(person_dir, '*.png'))
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1, 1))
                num_imgs_per_tracklet.append(len(img_names))
                #fid.write("cam2_" + dirname + '\n')

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

"""Create dataset"""

__factory = {
    'ilidsvid': iLIDSVID,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)

if __name__ == '__main__':
    # test
    dataset = iLIDSVID()

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


