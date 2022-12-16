import os
import os.path as osp
import sys
import time
import numpy as np
import pandas as pd
import collections
import random
import math
## For torch lib
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torch.nn.functional as F
## For Image lib
from PIL import Image
import pdb

def Get_MOT_DataLoader(seq_len=10, num_workers=8, pin_memory=False, test_batch = 32):
    dataset = MOT()

    transform_test = T.Compose([
        T.Resize((224, 112)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    g_dataset = VideoDataset(dataset.query, seq_len=seq_len, sample='dense', transform=transform_test)
    q_dataset = VideoDataset(dataset.query, seq_len=seq_len, sample='dense', transform=transform_test)
    galleryloader = DataLoader(
    g_dataset,
    batch_size=test_batch, shuffle=False, num_workers=num_workers,
    pin_memory=False, drop_last=False,
    )
    queryloader = DataLoader(
    q_dataset,
    batch_size=test_batch, shuffle=False, num_workers=num_workers,
    pin_memory=False, drop_last=False,
    )
    return  galleryloader, queryloader


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None , max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        # if self.sample == 'restricted_random':
        #     frame_indices = range(num)
        #     chunks = 
        #     rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        #     begin_index = random.randint(0, rand_end)


        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = frame_indices[begin_index:end_index]
            # print(begin_index, end_index, indices)
            if len(indices) < self.seq_len:
                indices=np.array(indices)
                indices = np.append(indices , [indices[-1] for i in range(self.seq_len - len(indices))])
            else:
                indices=np.array(indices)
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            # import pdb
            # pdb.set_trace()
            cur_index=0
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)

                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid

        elif self.sample == 'dense_subset':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.max_length - 1)
            begin_index = random.randint(0, rand_end)
            

            cur_index=begin_index
            frame_indices = [i for i in range(num)]
            indices_list=[]
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=frame_indices[cur_index:]
            # print(last_seq)
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            

            indices_list.append(last_seq)
            imgs_list=[]
            # print(indices_list , num , img_paths  )
            for indices in indices_list:
                if len(imgs_list) > self.max_length:
                    break 
                imgs = []
                for index in indices:
                    index=int(index)
                    img_path = img_paths[index]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                #imgs=imgs.permute(1,0,2,3)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        
        elif self.sample == 'intelligent_random':
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            imgs = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            #imgs=imgs.permute(1,0,2,3)
            return imgs, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

class MOT(object):
    root = '../../datasets/MARS/'

    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    #track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    #track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_name_path = osp.join(root, 'info/query_name.txt')

    def __init__(self, min_seq_len=0):
        self._check_before_run()

        # prepare meta data
        print('sorting names')
        train_names = sorted(self._get_names(self.train_name_path))
        print('sorting test names')
        test_names = sorted(self._get_names(self.test_name_path))
        #track_train = loadmat(self.track_train_info_path)['track_train_info'] # numpy.ndarray (8298, 4)
        #track_test = loadmat(self.track_test_info_path)['track_test_info'] # numpy.ndarray (12180, 4)
        query_names = sorted(self._get_names(self.query_name_path)) # numpy.ndarray (1980,)
        #query_IDX -= 1 # index from 0
        print("sorting gallery names", len(test_names))
        gallery_names = [name for name in test_names if name not in query_names]
        print('processing training data')
        train, num_train_tracklets, num_train_pids, num_train_imgs = \
          self._process_data(train_names, home_dir='bbox_test', relabel=True, min_seq_len=min_seq_len)
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))

        print('processing query data')
        query, num_query_tracklets, num_query_pids, num_query_imgs = \
          self._process_data(query_names, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)
        print('processing gallery data')
        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
          self._process_data(gallery_names, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_query_imgs + num_gallery_imgs + num_train_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_gallery_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MOT loaded")
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

        # self.train_videos = video
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids + 1
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        #if not osp.exists(self.train_name_path):
        #    raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        #if not osp.exists(self.track_train_info_path):
        #    raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        #if not osp.exists(self.track_test_info_path):
        #    raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_name_path):
            raise RuntimeError("'{}' is not available".format(self.query_name_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = len(os.listdir(self.root + home_dir))
        pid_list = os.listdir(self.root + home_dir)
        num_pids = len(pid_list)

        #if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        #txt_name = self.root + home_dir + str(len(meta_data)) + '.txt'
        #fid = open(txt_name, "w")
        start_index = 0
        end_index = 0
        max_pid = 0
        for tracklet_idx in sorted(os.listdir(self.root + home_dir)):
            for i in range(start_index, len(names)):
                if end_index > (len(names) - 1) or tracklet_idx + "F" not in names[end_index]:
                    if start_index == end_index:
                        break
                    else:
                        #if relabel: pid = pid2label[pid]
                        img_names = names[start_index:end_index]
                        if len(img_names) == 0:
                            break
                        # make sure image names correspond to the same person
                        pnames = [img_name[:4] for img_name in img_names]
                        assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

                        # append image names with directory information
                        img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
                        if len(img_paths) >= min_seq_len:
                            img_paths = tuple(img_paths)
                            if int(pnames[0]) > max_pid:
                                max_pid = int(pnames[0])
                            tracklets.append((img_paths, int(pnames[0]), 1))
                            num_imgs_per_tracklet.append(len(img_paths))
                            #fid.write(img_names[0] + '\n')
                        start_index = end_index
                else:
                    end_index += 1
                    #fid.close()
        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, max_pid, num_imgs_per_tracklet





def process_labels(labels):
    unique_id = np.unique(labels)
    id_count = len(unique_id)
    id_dict = {ID:i for i,  ID in enumerate(unique_id.tolist())}
    for i in range(len(labels)):
        labels[i] = id_dict[labels[i]]
    assert len(unique_id)-1 == np.max(labels)
    return labels, id_count

class Video_train_Dataset(Dataset):
    def __init__(self, db_txt, info, transform, seq_len=6, track_per_class=4, flip_p=0.5,
                 delete_one_cam=False, cam_type='normal'):

        with open(db_txt, 'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))

        # For info (id, track)
        if delete_one_cam == True:
            info = np.load(info)
            info[:, 2], id_count = process_labels(info[:, 2])
            for i in range(id_count):
                idx = np.where(info[:, 2]==i)[0]
                if len(np.unique(info[idx, 3])) ==1:
                    info = np.delete(info, idx, axis=0)
                    id_count -=1
            info[:, 2], id_count = process_labels(info[:, 2])
            #change from 625 to 619
        else:
            info = np.load(info)
            info[:, 2], id_count = process_labels(info[:, 2])


        self.info = []
        for i in range(len(info)):
            sample_clip = []
            F = info[i][1]-info[i][0]+1 # frames
            if F == 0:
                continue

            if F < seq_len:
                strip = list(range(info[i][0], info[i][1]+1))+[info[i][1]]*(seq_len-F)
                for s in range(seq_len):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/seq_len)
                strip = list(range(info[i][0], info[i][1]+1))+[info[i][1]]*(interval*seq_len-F)
                for s in range(seq_len):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][4]]))

        #pdb.set_trace()
        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = id_count
        self.n_tracklets = self.info.shape[0]
        self.flip_p = flip_p
        self.track_per_class = track_per_class
        self.cam_type = cam_type
        self.two_cam = False
        self.cross_cam = False

    def __getitem__(self, ID):
        sub_info = self.info[self.info[:, 1] == ID]

        if self.cam_type == 'normal':
            tracks_pool = list(np.random.choice(sub_info[:, 0], self.track_per_class))
        elif self.cam_type == 'two_cam':
            unique_cam = np.random.permutation(np.unique(sub_info[:, 2]))[:2]
            tracks_pool = list(np.random.choice(sub_info[sub_info[:, 2]==unique_cam[0], 0], 1))+\
                list(np.random.choice(sub_info[sub_info[:, 2]==unique_cam[1], 0], 1))
        elif self.cam_type == 'cross_cam':
            unique_cam = np.random.permutation(np.unique(sub_info[:, 2]))
            while len(unique_cam) < self.track_per_class:
                unique_cam = np.append(unique_cam, unique_cam)
            unique_cam = unique_cam[:self.track_per_class]
            tracks_pool = []
            for i in range(self.track_per_class):
                tracks_pool += list(np.random.choice(sub_info[sub_info[:, 2]==unique_cam[i], 0], 1))

        one_id_tracks = []
        for track_pool in tracks_pool:
            idx = np.random.choice(track_pool.shape[1], track_pool.shape[0])
            number = track_pool[np.arange(len(track_pool)), idx]
            imgs = [self.transform(Image.open(path)) for path in self.imgs[number]]
            imgs = torch.stack(imgs, dim=0)

            random_p = random.random()
            if random_p < self.flip_p:
                imgs = torch.flip(imgs, dims=[3])
            one_id_tracks.append(imgs)
        return torch.stack(one_id_tracks, dim=0),  ID*torch.ones(self.track_per_class, dtype=torch.int64)

    def __len__(self):
        return self.n_id

def Video_train_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key, value in zip(data[0].keys(), values)}
    else:
        imgs, labels = zip(*data)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(labels, dim=0)
        return imgs, labels

def Get_Video_train_DataLoader(db_txt, info, transform, shuffle=True, num_workers=8, seq_len=10,
                               track_per_class=4, class_per_batch=8):
    dataset = Video_train_Dataset(db_txt, info, transform, seq_len, track_per_class)
    dataloader = DataLoader(
        dataset, batch_size=class_per_batch, collate_fn=Video_train_collate_fn, shuffle=shuffle,
        worker_init_fn=lambda _:np.random.seed(), drop_last=True, num_workers=num_workers)
    return dataloader

class Video_test_rrs_Dataset(Dataset):
    def __init__(self, db_txt, info, query, transform, seq_len=6, distractor=True):
        with open(db_txt, 'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # info
        info = np.load(info)
        self.info = []
        for i in range(len(info)):
            if distractor == False and info[i][2]==0:
                continue
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F == 0:
                continue

            if F < seq_len:
                strip = list(range(info[i][0], info[i][1]+1))+[info[i][1]]*(seq_len-F)
                for s in range(seq_len):
                    pool = strip[s*1:(s+1)*1]
                    sample_clip.append(list(pool))
            else:
                interval = math.ceil(F/seq_len)
                strip = list(range(info[i][0], info[i][1]+1))+[info[i][1]]*(interval*seq_len-F)
                for s in range(seq_len):
                    pool = strip[s*interval:(s+1)*interval]
                    sample_clip.append(list(pool))
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][4]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = len(np.unique(self.info[:, 1]))
        self.n_tracklets = self.info.shape[0]
        self.query_idx = np.load(query).reshape(-1)

        if distractor == False:
            zero = np.where(info[:, 2]==0)[0]
            self.new_query = []
            for i in self.query_idx:
                if i < zero[0]:
                    self.new_query.append(i)
                elif i <= zero[-1]:
                    continue
                elif i > zero[-1]:
                    self.new_query.append(i-len(zero))
                else:
                    continue
            self.query_idx = np.array(self.new_query)

    def __getitem__(self, idx):
        clips = self.info[idx, 0]
        imgs = [self.transform(Image.open(path)) for path in self.imgs[clips[:, 0]]]
        imgs = torch.stack(imgs, dim=0)
        label = self.info[idx, 1]*torch.ones(1, dtype=torch.int32)
        cam = self.info[idx, 2]*torch.ones(1, dtype=torch.int32)
        paths = [path for path in self.imgs[clips[:, 0]]]
        paths = np.stack(paths, axis=0)
        return imgs, label, cam, paths
    def __len__(self):
        return len(self.info)

def Video_test_rrs_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key, value in zip(data[0].keys(), values)}
    else:
        imgs, label, cam, paths, oids= zip(*data)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(label, dim=0)
        cams = torch.cat(cam, dim=0)
        paths = np.concatenate(paths, axis=0)
        oids = torch.cat(oids, dim=0)
        return imgs, labels, cams, paths, oids

def Get_Video_test_rrs_DataLoader(db_txt, info, query, transform, batch_size=10, shuffle=False,
                              num_workers=8, seq_len=6, distractor=True):
    dataset = Video_test_rrs_Dataset(db_txt, info, query, transform, seq_len, distractor=distractor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=Video_test_rrs_collate_fn, shuffle=shuffle,
        worker_init_fn=lambda _:np.random.seed(), num_workers=num_workers)
    return dataloader

class Video_test_all_Dataset(Dataset):
    def __init__(self, db_txt, info, query, transform, seq_len=6, distractor=True):
        with open(db_txt, 'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))
        # info
        info = np.load(info)
        self.info = []
        for i in range(len(info)):
            if distractor == False and info[i][2]==0:
                continue
            sample_clip = []
            F = info[i][1]-info[i][0]+1
            if F == 0:
                continue
            strip = list(range(info[i][0], info[i][1]+1))
            sample_clip.append(strip)
            
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][4], info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = len(np.unique(self.info[:, 1]))
        self.n_tracklets = self.info.shape[0]
        self.query_idx = np.load(query).reshape(-1)

        if distractor == False:
            zero = np.where(info[:, 2]==0)[0]
            self.new_query = []
            for i in self.query_idx:
                if i < zero[0]:
                    self.new_query.append(i)
                elif i <= zero[-1]:
                    continue
                elif i > zero[-1]:
                    self.new_query.append(i-len(zero))
                else:
                    continue
            self.query_idx = np.array(self.new_query)

    def __getitem__(self, idx):
        clips = self.info[idx, 0]
        imgs = [self.transform(Image.open(path)) for path in self.imgs[clips[0]]]
        imgs = torch.stack(imgs, dim=0)
        label = self.info[idx, 1]*torch.ones(1, dtype=torch.int32)
        cam = self.info[idx, 2]*torch.ones(1, dtype=torch.int32)
        paths = [path for path in self.imgs[clips[0]]]
        paths = np.stack(paths, axis=0)
        oids = self.info[idx, 3]*torch.ones(1, dtype=torch.int32)
        return imgs, label, cam, paths, oids

    def __len__(self):
        return len(self.info)

def Video_test_all_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data)
        return {key:value  for key, value in zip(data[0].keys(), values)}
    else:
        imgs, label, cam, paths, oids = zip(*data)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(label, dim=0)
        cams = torch.cat(cam, dim=0)
        paths = np.concatenate(paths, axis=0)
        oids = torch.cat(oids, dim=0)
        return imgs, labels, cams, paths, oids

def Get_Video_test_all_DataLoader(db_txt, info, query, transform, batch_size=10, shuffle=False,
                              num_workers=8, seq_len=6, distractor=True):
    dataset = Video_test_all_Dataset(db_txt, info, query, transform, seq_len, distractor=distractor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=Video_test_all_collate_fn, shuffle=shuffle,
        worker_init_fn=lambda _:np.random.seed(), num_workers=num_workers)
    return dataloader

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img
