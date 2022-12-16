from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import numpy as np
import os.path as osp
import random
import copy

from PIL import Image
from collections import defaultdict
from bisect import bisect_right
from itertools import permutations

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchreid.transforms import ImageData
from torchreid.utils.reidtools import calc_splits


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all', 'consecutive', 'dense', 'restricted', 'skipdense']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, training=False, pose_info=None,
                 num_split=8, num_parts=3, num_scale=True, pyramid_part=True, enable_pose=True, max_len=1000):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.training = training
        self.pose_info = pose_info
        self.num_split = num_split
        self.num_parts = num_parts
        self.num_scale = num_scale
        self.pyramid_part = pyramid_part
        self.enable_pose = enable_pose
        # XXX: make sure max_len not too big and divided by seq_len
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid, oid = self.dataset[index]
        num = len(img_paths)
        # abandon the over length images
        if num > self.max_len:
            num = self.max_len
            img_paths = img_paths[:num]

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num / self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        elif self.sample == 'consecutive':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = np.arange(num)
            rand_end = max(0, num - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, num)

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len: break
                np.append(indices, index)
        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, 
            batch_size needs to be set to 1. This sampling strategy is used in test phase.
            """
            indices = np.arange(num)
            append_size = self.seq_len - num % self.seq_len
            indices = np.append(indices, [num - 1] * append_size)
        elif self.sample == 'restricted':
            total_indices = np.arange(num)
            append_size = self.seq_len - num % self.seq_len
            total_indices = np.append(total_indices, [num - 1] * append_size)

            chunk_size = len(total_indices) // self.seq_len
            indices = []
            for seq_idx in range(self.seq_len):
                chunk_index = total_indices[seq_idx * chunk_size: (seq_idx + 1) * chunk_size]
                idx = np.random.choice(chunk_index, 1)
                indices.append(idx)
            indices = np.sort(indices)
        elif self.sample == 'skipdense':
            """
            Sample all frames in the video into a list of frames, and frame index is increased by video_len / seq_len.
            """
            indices = np.arange(num)
            append_size = self.seq_len - num % self.seq_len
            indices = np.append(indices, [num - 1] * append_size)
            skip_len = len(indices) // self.seq_len
            final_indices = []
            for i in range(skip_len):
                final_indices.extend([indices[idx] for idx in np.arange(i, len(indices), skip_len)])
            indices = final_indices
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        img_sizes = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            img_sizes.append(img.size)
            imgs.append(ImageData(img))

        if self.transform is not None:
            imgs = self.transform(imgs)
            imgs = [img.img for img in imgs]

        # generate pose related graph
        assert isinstance(self.pose_info, dict), 'please load the pose info'
        if self.sample in ['dense', 'skipdense']:
            if self.enable_pose:
                adj_list = []
                for i in range(len(indices) // self.seq_len):
                    cur_indices = indices[i * self.seq_len: (i + 1) * self.seq_len]
                    cur_adj = generate_graph(imgs[i * self.seq_len: (i + 1) * self.seq_len],
                                             im_paths=[img_paths[int(index)] for index in cur_indices],
                                             im_sizes=img_sizes, poses=self.pose_info,
                                             num_split=self.num_split, num_parts=self.num_parts,
                                             num_scale=self.num_scale, pyramid_part=self.pyramid_part)
                    adj_list.append(cur_adj)
                adj = torch.stack(adj_list, dim=0)
            else:
                adj_size = sum(calc_splits(self.num_split)) if self.pyramid_part else self.num_split
                adj_size = adj_size * self.seq_len * self.num_scale
                adj = torch.ones((len(indices) // self.seq_len, adj_size, adj_size))
            imgs = torch.stack(imgs, dim=0)
            imgs = imgs.view(-1, self.seq_len, imgs.size(1), imgs.size(2), imgs.size(3))
        else:
            if self.enable_pose:
                adj = generate_graph(imgs, im_paths=[img_paths[int(index)] for index in indices], im_sizes=img_sizes,
                                     poses=self.pose_info, num_split=self.num_split, num_parts=self.num_parts,
                                     num_scale=self.num_scale, pyramid_part=self.pyramid_part)
            else:
                adj_size = sum(calc_splits(self.num_split)) if self.pyramid_part else self.num_split
                adj_size = adj_size * self.seq_len * self.num_scale
                adj = torch.ones((adj_size, adj_size))
            imgs = torch.stack(imgs, dim=0)

        return imgs, pid, camid, oid, adj


def generate_graph(ims, im_paths, im_sizes, poses, num_split, num_parts, num_scale, pyramid_part,
                   threshold=0.1):
    """create pose-guided graph
    {0,  "Nose"}, {1,  "Neck"}, {2,  "RShoulder"}, {3,  "RElbow"}, {4,  "RWrist"},
    {5,  "LShoulder"}, {6,  "LElbow"}, {7,  "LWrist"}, {8,  "RHip"}, {9,  "RKnee"},
    {10, "RAnkle"}, {11, "LHip"}, {12, "LKnee"}, {13, "LAnkle"}, {14, "REye"},
    {15, "LEye"}, {16, "REar"}, {17, "LEar"}"""
    part_contain_list = []
    for im, path, size in zip(ims, im_paths, im_sizes):
        """
        ilidsvid
            key format: 
            cam2_person188_03266.png
            path:
            ['data/ilids-vid/i-LIDS-VID/sequences/cam1/person238/cam1_person238_02519.png']
        mars (there are no id overlap between train and test folders)
            key format: 
            0999C1T0001F002.jpg
            path:
            ['data/mars/bbox_train/0999/0999C1T0001F002.jpg']
        prid
            key format: 
            cam_a-person_0115-0006.png
            path:
            ['data/prid2011/prid_2011/multi_shot/cam_a/person_0115/0006.png']
        dukemtmc-vidreid
            key format:
            0148-0212-0148_C5_F0006_X89499.jpg
            path:
            ['data/dukemtmc-vidreid/DukeMTMC-VideoReID/train/0148/0212/0148_C5_F0006_X89499.jpg']
        """
        if 'ilids-vid' in path:  # ilidsvid
            key = path.split('/')[-1]
        elif 'prid2011' in path:  # prid2011
            key = '-'.join(path.split('/')[-3:])
        elif 'mars' in path:  # mars
            key = path.split('/')[-1]
        elif 'duke' in path:  # dukemtmcvidreid
            key = '-'.join(path.split('/')[-3:])
        else:
            raise ValueError('{} is not acceptable'.format(path))

        # import matplotlib.pyplot as plt
        # from skimage.draw import circle, line
        # temp_img = (im.permute(1, 2, 0).numpy() * 255).astype(int)
        # splits = np.arange(0, size[1], size[1] / num_parts)
        # # for s in splits:
        # #     temp_img[int(s * im.size(2) / size[0]), :, :] = 0
        # pose_img = np.ones_like(temp_img) * 255
        # plt.imsave('pose_sample/slice_{}'.format(key), temp_img / 255)
        #
        # try:
        #     temp_pose = poses[key]
        #     try:
        #         skeleton = [[0, 1], [0, 14], [0, 15], [14, 15], [15, 16], [14, 17],  # head
        #                     [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],  # body
        #                     [8, 9], [9, 10], [11, 12], [12, 13]]  # leg
        #         for pair in skeleton:
        #             p1 = temp_pose[pair[0]]
        #             p2 = temp_pose[pair[1]]
        #             if p1[2] > threshold and p2[2] > threshold:
        #                 x1, y1 = int(p1[0] * im.size(2) / size[0]), int(p1[1] * im.size(1) / size[1])
        #                 x2, y2 = int(p2[0] * im.size(2) / size[0]), int(p2[1] * im.size(1) / size[1])
        #                 rr, cc = line(y1, x1 - 1, y2, x2 - 1)
        #                 temp_img[rr, cc] = np.array([255, 255, 0])
        #                 pose_img[rr, cc] = np.array([255, 255, 0])
        #
        #         for idx, kp in enumerate(temp_pose):
        #             tmp_kp = [int(kp[0] * im.size(2) / size[0]),
        #                       int(kp[1] * im.size(1) / size[1])]
        #             rr, cc = circle(tmp_kp[1], tmp_kp[0], radius=2)
        #             if kp[2] > threshold:
        #                 if idx in [0, 14, 15, 16, 17]:
        #                     color = np.array([255, 0, 0])  # red head
        #                 elif idx in [1, 2, 3, 4, 5, 6, 7]:
        #                     color = np.array([0, 255, 0])  # green body
        #                 else:  # idx in [8, 9, 10, 11, 12, 13]:
        #                     color = np.array([0, 0, 255])  # blue leg
        #                 temp_img[rr, cc] = color
        #                 pose_img[rr, cc] = color
        #         plt.imsave('pose_sample/combine_{}'.format(key), temp_img / 255)
        #         # plt.imsave('pose_sample/pose_{}'.format(key), pose_img / 255)
        #         # plt.imsave('pose_sample/origin_{}'.format(key), (im.permute(1, 2, 0).numpy() * 255).astype(int) / 255)
        #     except:
        #         print('some error in keypoint plot in {}'.format(key))
        # except:
        #     print('{} not detected'.format(key))
        #     plt.imsave('pose_sample/combine_{}'.format(key), temp_img / 255)
        #     # plt.imsave('pose_sample/pose_{}'.format(key), pose_img / 255)
        #     # plt.imsave('pose_sample/origin_{}'.format(key), (im.permute(1, 2, 0).numpy() * 255).astype(int) / 255)
        splits = np.arange(0, size[1] + 1, size[1] / num_split)
        part_contain = defaultdict(set)
        try:
            tmp_pose = poses[key]
            # if get the pose
            # nose, neck, reye, leye, rear, lear is head part
            # rshoulder, relbow, rwrist, lshoulder, lelbow, lwrist is body part
            # rhip, rknee, rankle, lhip, lknee, lankle is leg part
            body_id_dict = {'head': [0, 1, 14, 15, 16, 17],
                            'body': [2, 3, 4, 5, 6, 7],
                            'leg': [8, 9, 10, 11, 12, 13]}
            for part_name, part_ids in body_id_dict.items():
                for p_id in part_ids:
                    if tmp_pose[p_id, 2] > threshold:
                        # confidence is over threshold
                        # XXX: splits, start 1 end num_split
                        loc_split_id = bisect_right(splits, tmp_pose[p_id, 1])
                        loc_split_id = min(num_split, max(1, loc_split_id))
                        part_contain[part_name].add(loc_split_id)
            for part_name, part_contain_set in part_contain.items():
                # let the split contained in each part be continous
                if len(part_contain_set) > 1:
                    new_part_contain_set = set(list(range(min(part_contain_set), max(part_contain_set) + 1)))
                    part_contain[part_name].update(new_part_contain_set)
        except:  # if no person detected, let part_contain empty
            pass
        part_contain_list.append(part_contain)
    try:
        adj = adj_graph(part_contain_list, num_parts=num_parts, num_split=num_split, pyramid_part=pyramid_part,
                        method='same')
        adj = create_multiscale_graph(adj, num_scale=num_scale)
    except:
        raise RuntimeError

    return adj


def adj_graph(part_contain_list, num_parts, num_split, pyramid_part, method='adjacent'):
    if num_parts == 3:
        part_names = ['head', 'body', 'leg']
    else:
        raise NotImplementedError

    seq_len = len(part_contain_list)
    num_total_splits = sum(calc_splits(num_split)) if pyramid_part else num_split

    if pyramid_part:
        # extend part_contain_list
        k = int(np.log2(num_split))
        new_part_contain_list = copy.deepcopy(part_contain_list)
        # print('part_contain_list: {}'.format(part_contain_list))
        for idx, part_contain_i in enumerate(part_contain_list):
            for part_name, part_contain_set in part_contain_i.items():
                new_set = new_part_contain_list[idx][part_name]
                cur_set = part_contain_set
                for split_id in cur_set:
                    # generate the new split id, e.g. if num_split == 8, {1} -> {1, 9, 13, 15} and {3} -> {2, 9, 13, 15}
                    new_set.update([int(np.ceil(split_id / np.power(2, i))) + (np.power(2, k+1) - np.power(2, k+1-i))
                                    for i in range(1, k + 1)])
                new_part_contain_list[idx][part_name] = new_set
        part_contain_list = new_part_contain_list
        # print('new_part_contain_list: {}'.format(new_part_contain_list))

    adj = torch.zeros((num_total_splits * seq_len, num_total_splits * seq_len))
    part_name_id_pairs = [[i, i] for i in range(num_parts)]
    if method == 'adjacent':
        # link up adajcent part corresponde splits
        part_name_id_pairs += [[i, i+1] for i in range(num_parts - 1)]
    part_name_pairs = [(part_names[id0], part_names[id1]) for id0, id1 in part_name_id_pairs]

    for part_name_pair in part_name_pairs:
        related_split_set = set()
        for seq_id in range(seq_len):
            related_split_set.update(
                [split_id + seq_id * num_total_splits for split_id in part_contain_list[seq_id][part_name_pair[0]]])
            if part_name_pair[0] != part_name_pair[1]:
                related_split_set.update(
                    [split_id + seq_id * num_total_splits for split_id in part_contain_list[seq_id][part_name_pair[1]]])
        for related_pair in permutations(related_split_set, 2):
            adj[related_pair[0] - 1, related_pair[1] - 1] = 1
    return adj


def create_multiscale_graph(adj, num_scale=3):
    if num_scale == 1:
        return adj
    # create multi scale adjacent matrix
    size = adj.size(1)
    new_adj = torch.zeros((num_scale * size, num_scale * size))
    I = torch.eye(size)
    for si in range(num_scale):
        for sj in range(num_scale):
            if si == sj:
                new_adj[si * size: (si + 1) * size, sj * size: (sj + 1) * size] = adj
            else:
                new_adj[si * size: (si + 1) * size, sj * size: (sj + 1) * size] = I
    return new_adj
