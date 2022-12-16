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
'''
For MARS, Video-based Re-ID
'''

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
                 delete_one_cam=False, cam_type='cross_cam'):

        with open(db_txt, 'r') as f:
            self.imgs = np.array(f.read().strip().split('\n'))

        # For info (id, track)
        if delete_one_cam == True:
            info = np.load(info)
            info[:, 2], id_count = process_labels(info[:, 2])
            for i in range(id_count):
                idx = np.where(info[:, 2]==i)[0]
                if len(np.unique(info[idx, 4])) ==1:
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
            F = info[i][1]-info[i][0]+1
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
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][4], info[i][3]]))

        self.info = np.array(self.info)
        self.transform = transform
        self.n_id = id_count
        self.n_cam = len(np.unique(self.info[:, 2]))
        self.flip_p = flip_p
        self.track_per_class = track_per_class
        self.cam_type = cam_type

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

        return_imgs = torch.stack(one_id_tracks, dim=0)
        return_labels = ID*torch.ones(self.track_per_class, dtype=torch.int64)
        return_cams = torch.from_numpy(np.array(unique_cam, dtype=np.long)) - 1

        return return_imgs, return_labels, return_cams

    def __len__(self):
        return self.n_id

def Video_train_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        values = MARS_collate_fn(t_data) ## weird
        return {key:value  for key, value in zip(data[0].keys(), values)}
    else:
        imgs, labels, cams, oids = zip(*data)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(labels, dim=0)
        cams = torch.cat(cams, dim=0)
        oids = torch.cat(oids, dim=0)
        return imgs, labels, cams, oids

def Get_Video_train_DataLoader(db_txt, info, transform, shuffle=True, num_workers=8, seq_len=10,
                               track_per_class=4, class_per_batch=8):
    dataset = Video_train_Dataset(db_txt, info, transform, seq_len, track_per_class)
    dataloader = DataLoader(
        dataset, batch_size=class_per_batch, collate_fn=Video_train_collate_fn, shuffle=shuffle,
        worker_init_fn=lambda _:np.random.seed(), drop_last=True, num_workers=num_workers)
    return dataloader

class Video_test_Dataset(Dataset):
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
            self.info.append(np.array([np.array(sample_clip), info[i][2], info[i][3]]))

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
        cam = self.info[idx, 2]*torch.ones(1, dtype=torch.int64)
        oids = self.info[idx, 3]*torch.ones(1, dtype=torch.int32)
        paths = [path for path in self.imgs[clips[:, 0]]]
        paths = np.stack(paths, axis=0)

        return imgs, label, cam, paths, oids

    def __len__(self):
        return len(self.info)

def Video_test_collate_fn(data):
    if isinstance(data[0], collections.Mapping):
        t_data = [tuple(d.values()) for d in data]
        print("here")
        values = MARS_collate_fn(t_data)
        return {key:value  for key, value in zip(data[0].keys(), values)}
    else:
        print("there")
        imgs, label, cam, paths, oids = zip(*data)
        imgs = torch.cat(imgs, dim=0)
        labels = torch.cat(label, dim=0)
        cams = torch.cat(cam, dim=0)
        paths = np.concatenate(paths, axis=0)
        oids = torch.cat(oids, dim=0)
        return imgs, labels, cams, paths, oids

def Get_Video_test_DataLoader(db_txt, info, query, transform, batch_size=10, shuffle=False,
                              num_workers=8, seq_len=6, distractor=True):
    dataset = Video_test_Dataset(db_txt, info, query, transform, seq_len, distractor=distractor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=Video_test_collate_fn, shuffle=shuffle,
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
