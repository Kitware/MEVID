from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import torch.backends.cudnn as cudnn

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torchvision.transforms as transforms
# from tools import * 
import models

from loss import CrossEntropyLabelSmooth, TripletLoss , CenterLoss , OSM_CAA_Loss
from tools.transforms2 import *
from tools.scheduler import WarmupMultiStepLR
from tools.utils import AverageMeter, Logger, save_checkpoint , resume_from_checkpoint
from tools.eval_metrics import evaluate
from tools.samplers import RandomIdentitySampler
from tools.video_loader import VideoDataset , VideoDataset_inderase
import tools.data_manager as data_manager

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())

# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-f', '--focus', type=str, default='map', help="map,rerank_map")
parser.add_argument('--feat-dir', type=str, default="/home/local/KHQ/alexander.lynch/DIY-AI/person_reid/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them/models/log/")

args = parser.parse_args()

torch.manual_seed(args.seed)

# args.gpu_devices = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()

cudnn.benchmark = True
torch.cuda.manual_seed_all(args.seed)


from tools.eval_metrics import evaluate , re_ranking
def test_rerank(feat_dir, ranks=[1, 5, 10, 20]):
    query = torch.load(osp.join(feat_dir, 'query_feat.pth'))
    qf = query['qf']
    q_pids = query["q_pids"]
    q_camids = query["q_camids"]
    print(type(qf), qf.size(0))
    gallery1 = torch.load(osp.join(feat_dir, 'gallery_feat_1.pth'))
    gallery2 = torch.load(osp.join(feat_dir, 'gallery_feat_2.pth'))
    gallery3 = torch.load(osp.join(feat_dir, 'gallery_feat_3.pth'))

    gallery = {}
    for key in gallery1.keys():
        print(key)
        if key == "gf":
            tensor = torch.cat((gallery1[key], gallery2[key], gallery3[key]), dim=0)
            gallery[key] = tensor
        else:
            value = np.concatenate((gallery1[key], gallery2[key], gallery3[key]), axis=None)
            gallery[key] = value    
    
    gf = gallery['gf']
    g_pids = gallery["g_pids"]
    g_camids = gallery["g_camids"]

    print("Computing distance matrix")
    print(gf.size, gf.shape, type(g_pids), type(g_camids))
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    distmat_rerank = re_ranking(qf,gf)
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Rerank Computing CMC and mAP")
    re_rank_cmc, re_rank_mAP = evaluate(distmat_rerank, q_pids, g_pids, q_camids, g_camids)
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print("Results ---------- ")
    if 'mars' in args.dataset :
        print("mAP: {:.1%} vs {:.1%}".format(mAP, re_rank_mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%} vs {:.1%}".format(r, cmc[r-1], re_rank_cmc[r-1]))
    print("------------------")
    if 'mars' not in args.dataset :
        print("Dataset not MARS : instead", args.dataset)
        return cmc[0]
    else:
        if args.focus == "map":
            print("returning map")
            return mAP
        else:
            print("returning re-rank")
            return re_rank_mAP


for feat_dir in sorted(os.listdir(args.feat_dir)):
    if ".tar" not in feat_dir:
        print("EVALUATING ", feat_dir)
        test_rerank(args.feat_dir + feat_dir)