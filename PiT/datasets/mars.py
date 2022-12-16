import os.path as osp
from scipy.io import loadmat
import numpy as np

from .bases import BaseImageDataset
import pdb
class Mars(BaseImageDataset):
    """
    MARS

    Reference:
    Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.
    """

    def __init__(self, root='/data/datasets/', min_seq_len=0):
        self.root = osp.join(root, 'reid_final')
        self.train_name_path = osp.join(self.root, 'train_name.txt')
        self.test_name_path = osp.join(self.root, 'test_name.txt')
        self.track_train_info_path = osp.join(self.root, 'track_train_info.txt')
        self.track_test_info_path = osp.join(self.root, 'track_test_info.txt')
        self.query_IDX_path = osp.join(self.root, 'query_IDX.txt')

        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = np.loadtxt(self.track_train_info_path).astype(np.int) # numpy.ndarray (2394, 4)
        track_test = np.loadtxt(self.track_test_info_path).astype(np.int) # numpy.ndarray (2394, 4)
        query_IDX = np.loadtxt(self.query_IDX_path).astype(np.int) # numpy.ndarray (1980,)
        #query_IDX -= 1 # index from 0
        track_query = track_test[query_IDX,:]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX,:]
        # track_gallery = track_test

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        pid2label, cid2label = self._id2label_test(track_query, track_gallery)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len, pid2label=pid2label, cid2label=cid2label)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len, pid2label=pid2label, cid2label=cid2label)

        train_img, _, _ = \
          self._extract_1stfeame(train_names, track_train, home_dir='bbox_train', relabel=True)

        query_img, _, _ = \
          self._extract_1stfeame(test_names, track_query, home_dir='bbox_test', relabel=False)

        gallery_img, _, _ = \
          self._extract_1stfeame(test_names, track_gallery, home_dir='bbox_test', relabel=False)

        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        total_num = np.sum(num_imgs_per_tracklet)
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=> MEVID loaded")
        print("Dataset statistics:")
        print("  -----------------------------------------")
        print("  subset   | # ids | # tracklets | # images")
        print("  -----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:8d}".format(num_train_pids, num_train_tracklets, np.sum(num_train_imgs)))
        print("  query    | {:5d} | {:8d} | {:8d}".format(num_query_pids, num_query_tracklets, np.sum(num_query_imgs)))
        print("  gallery  | {:5d} | {:8d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets, np.sum(num_gallery_imgs)))
        print("  -----------------------------------------")
        print("  total    | {:5d} | {:8d} | {:8d}".format(num_total_pids, num_total_tracklets, total_num))
        print("  -----------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  -----------------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.train_img = train_img
        self.query_img = query_img
        self.gallery_img = gallery_img

        self.pid2label = pid2label
        self.cid2label = cid2label

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _id2label_test(self, query_info, gallery_info):
        pid_container = set()
        cid_container = set()
        num_query = query_info.shape[0]
        for query_idx in range(num_query):
            _, _, pid, _, cid = query_info[query_idx,...]
            pid_container.add(pid)
            cid_container.add(cid)

        num_gallery = gallery_info.shape[0]
        for gallery_idx in range(num_gallery):
            _, _, pid, _, cid = gallery_info[gallery_idx,...]
            pid_container.add(pid)
            cid_container.add(cid)

        pid_container = sorted(pid_container)
        cid_container = sorted(cid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        cid2label = {cid:label for label, cid in enumerate(cid_container)}

        return pid2label, cid2label


    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0, pid2label=None, cid2label=None):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)
        cid_list = list(set(meta_data[:,4].tolist()))

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        if relabel: cid2label = {cid:label for label, cid in enumerate(cid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, oid, camid = data

            if pid == -1: continue # junk images are just ignored
            #assert 1 <= camid <= 6
            if pid2label is not None: pid = pid2label[pid]
            #camid -= 1 # index starts from 0
            if cid2label is not None: camid = cid2label[camid]
            img_names = names[start_index:end_index+1]
            
            if len(img_names) == 0:
                continue
            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[9:12] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                tracklets.append((img_paths, pid, camid, 1, oid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

    def _extract_1stfeame(self, names, meta_data, home_dir=None, relabel=False):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:,2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        imgs = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx,...]
            start_index, end_index, pid, oid, camid = data
            if pid == -1: continue # junk images are just ignored
            #assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            #camid -= 1 # index starts from 0
            img_name = names[start_index]

            # append image names with directory information
            img_path = osp.join(self.root, home_dir, img_name[:4], img_name)
            
            imgs.append(([img_path], pid, camid, oid))

        num_imgs = len(imgs)

        return imgs, num_imgs, num_pids
