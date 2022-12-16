from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
random.seed(1)

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
        img_paths, pid, camid, oid = self.dataset[index]
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
            return imgs, pid, camid, oid

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
            return imgs_array, pid, camid, oid

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
            return imgs_array, pid, camid, oid
        
        elif self.sample == 'intelligent_random':
            # frame_indices = range(num)
            indices = []
            each = max(num//seq_len,1)
            for  i in range(seq_len):
                if i != seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            print(len(indices))
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
            return imgs, pid, camid, oid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))




import math

# from tools import RandomErasing_synch
# import transforms as T

def Random2DTranslation_params(h =224 , w =112, p=0.5 ):
    if random.random() < p:
        return None
    new_width, new_height = int(round(w * 1.125)), int(round(h * 1.125)) 
    x_maxrange = new_width - w
    y_maxrange = new_height - h
    x1 = int(round(random.uniform(0, x_maxrange)))
    y1 = int(round(random.uniform(0, y_maxrange)))
    return   new_width ,  new_height , x1 , y1 
        
        
def Random2DTranslation( img,  params , height=224 , width=112 , interpolation=Image.BILINEAR ):
    if params == None:
        return img.resize((width, height), interpolation) 
    resized_img = img.resize((params[0], params[1]), interpolation)
    croped_img = resized_img.crop((params[2], params[3], params[2] + width, params[3] + height))
    return croped_img
        


def erase_specs():
    sl=0.02
    # sl = 0.005
    # sh = 0.08
    sh=0.4
    r1 = 0.3
    mean=(0.4914, 0.4822, 0.4465)
    size1 = 112
    size2 = 224
    for attempt in range(100):
            area = size1 * size2
            target_area = random.uniform(sl, sh) * area / 10
            aspect_ratio = random.uniform(r1, 1 / r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < size1 and h <  size2 :
                x1 = random.randint(0, size2 - h)
                y1 = random.randint(0, size1 - w)
                return x1 , h , y1 , w 
    return None


class VideoDataset_synch(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    def __init__(self, dataset, seq_len=15, transform=None, sample=None ):
        self.dataset = dataset
        self.seq_len = seq_len
        self.re = transforms.Resize((224, 112), interpolation=3)
        self.pad = transforms.Pad(10)
        self.convert = transforms.ToTensor()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.probability = 0.5
        self.mean = [0.485, 0.456, 0.406]
        
    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        frame_indices = range(num)
        rand_end = max(0, len(frame_indices) - self.seq_len - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.seq_len, len(frame_indices))

        indices = frame_indices[begin_index:end_index]

        for index in indices:
            if len(indices) >= self.seq_len:
                break
            indices.append(index)
        indices=np.array(indices)
        imgs = []

        flip = transforms.RandomHorizontalFlip(p=0)
        if random.uniform(0, 1) >= self.probability:
                flip = transforms.RandomHorizontalFlip(p=1)

        erase = None
        if random.uniform(0, 1) >= self.probability:
                erase = True

        param = Random2DTranslation_params()
        specs = erase_specs()

        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            img = read_image(img_path)
            
            img = self.re(img)
            img = flip(img)
            img = self.pad(img)
            img = Random2DTranslation(img , param )
            img=  self.convert(img)
            img = self.normalize(img)

            if erase  and specs:
                    img[0, specs[0]:specs[0] + specs[1], specs[2]:specs[2] + specs[3]] = self.mean[0]
                    img[1, specs[0]:specs[0] + specs[1], specs[2]:specs[2] + specs[3]] = self.mean[1]
                    img[2, specs[0]:specs[0] + specs[1], specs[2]:specs[2] + specs[3]] = self.mean[2]
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        #imgs=imgs.permute(1,0,2,3)
        return imgs, pid, camid






from tools.transforms2 import RandomErasing3


class VideoDataset_inderase(Dataset):
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
        self.erase = RandomErasing3(probability=0.5, mean=[0.485, 0.456, 0.406])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample != "intelligent":
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))
            indices = list(frame_indices[begin_index:end_index])
            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
        else:
            # frame_indices = range(num)
            indices = []
            each = max(num//self.seq_len,1)
            for  i in range(self.seq_len):
                if i != self.seq_len -1:
                    indices.append(random.randint(min(i*each , num-1), min( (i+1)*each-1, num-1)) )
                else:
                    indices.append(random.randint(min(i*each , num-1), num-1) )
            # print(len(indices), indices, num )
        imgs = []
        labels = []
        for index in indices:
            index=int(index)
            img_path = img_paths[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img , temp  = self.erase(img)
            labels.append(temp)
            img = img.unsqueeze(0)
            imgs.append(img)
        labels = torch.tensor(labels)
        imgs = torch.cat(imgs, dim=0)
        #imgs=imgs.permute(1,0,2,3)
        return imgs, pid, camid , labels

        
