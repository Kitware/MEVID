from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import math
import h5py
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import data_manager
from video_loader import VideoDataset
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import models
from utils import AverageMeter, Logger, save_checkpoint
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from evaluation import evaluation

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
# Augment
parser.add_argument('--seq_len', type=int, default=8, help="number of images to sample in a tracklet")
parser.add_argument('--sample_stride', type=int, default=4, help="stride of images to sample in a tracklet")
parser.add_argument('--test_frames', default=8, type=int, help='frames/clip for test')
# Optimization options
parser.add_argument('--distance', type=str, default='consine', help="euclidean or consine")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='BiCnet_TKS')
parser.add_argument('--save-dir', type=str, default='./results')
parser.add_argument('--resume', type=str, default='', metavar='PATH')
# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        #cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset)

    # Data augmentation
    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_test = None

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids)
    model.cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        test_all_frames(model, queryloader, galleryloader, use_gpu)

    if not args.resume:
        for epoch in [150, 140, 130, 120]:
            weights = os.path.join(args.save_dir, 'checkpoint_ep'+str(epoch)+'.pth.tar')
            if not os.path.isfile(weights):
                continue
            print("Loading checkpoint from {}".format(weights))
            checkpoint = torch.load(weights)
            model.load_state_dict(checkpoint['state_dict'])
            test_all_frames(model, queryloader, galleryloader, use_gpu)


def test_all_frames(model, queryloader, galleryloader, use_gpu):
    model = nn.DataParallel(model)
    model.eval()
    with torch.no_grad():
        evaluation(model, args, queryloader, galleryloader, use_gpu)


if __name__ == '__main__':
    main()
