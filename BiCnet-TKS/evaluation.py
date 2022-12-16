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
from eval_metrics import evaluate, evaluate_clothes, evaluate_locations, evaluate_scales


def extract(model, args, vids, use_gpu):
    n, c, t, h, w = vids.size()
    assert(n == 1)
    k = args.test_frames

    if t % k != 0:
        inputs = vids.clone()
        while(inputs.size(2) % k != 0):
            for idx in range(t):
                if (inputs.size(2) % k == 0):
                    break
                inputs = torch.cat((inputs, vids[:,:,idx:idx+1]), 2)
        vids = inputs
    t = vids.size(2)
    assert (t % k == 0)

    vids = vids.view(c, t//k, k, h, w).contiguous()
    vids = vids.transpose(0, 1) #[t//k, c, k, h, w]
    vids = vids.cuda()

    num_clips = vids.size(0)
    batch_size = 32
    feat = torch.cuda.FloatTensor()
    for i in range(int(math.ceil(num_clips * 1.0 / batch_size))):
        clip = vids[i*batch_size: (i+1)*batch_size] #[batch_size, c, k, h, w]
        output = model(clip) #[batch_size, k/1, c]
        output = output.view(-1, output.size(-1)) #[batch_size*k, c]
        feat = torch.cat((feat, output), 0)

    feat = feat.mean(0, keepdim=True)
    feat = model.module.bn(feat)

    return feat

def evaluation(model, args, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    since = time.time()
    '''model.eval()

    qf, q_pids, q_camids, q_oids = [], [], [], []
    for batch_idx, (vids, pids, camids, oids) in enumerate(queryloader):
        if (batch_idx + 1) % 1000==0:
            print("{}/{}".format(batch_idx+1, len(queryloader)))

        qf.append(extract(model, args, vids, use_gpu).squeeze())
        q_pids.extend(pids)
        q_camids.extend(camids)
        q_oids.extend(oids)

    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    q_oids = np.asarray(q_oids)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids, g_oids = [], [], [], []
    for batch_idx, (vids, pids, camids, oids) in enumerate(galleryloader):
        if (batch_idx + 1) % 1000==0:
            print("{}/{}".format(batch_idx+1, len(galleryloader)))

        gf.append(extract(model, args, vids, use_gpu).squeeze())
        g_pids.extend(pids)
        g_camids.extend(camids)
        g_oids.extend(oids)

    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    g_oids = np.asarray(g_oids)'''

    '''if args.dataset == 'mars' or args.dataset == 'lsvid':
        print('process the dataset {}!'.format(args.dataset))
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)
        g_oids = np.append(q_oids, g_oids)'''

    '''print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))

    if args.distance == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
    else:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        distmat = - torch.mm(qf, gf.t())
    distmat = distmat.data.cpu()
    distmat = distmat.numpy()


    np.savez('eval.npz', distmat=distmat, q_pids=q_pids, g_pids=g_pids, q_camids=q_camids, g_camids=g_camids, q_oids=q_oids, g_oids=g_oids)'''
    data = np.load('eval.npz')
    distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids = data['distmat'], data['q_pids'], data['g_pids'], data['q_camids'], data['g_camids'], data['q_oids'], data['g_oids']

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Overall Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='SC')
    print("Same Clothes Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='CC')
    print("Changing Clothes Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL')
    print("Same Locations Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='DL')
    print("Changing Locations Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS')
    print("Same Scales Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='DS')
    print("Changing Scales Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    elapsed = round(time.time() - since)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}.".format(elapsed))

    return cmc[0]
