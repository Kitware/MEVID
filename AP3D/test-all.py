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
import pdb

import models
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import tools.data_manager as data_manager
from tools.video_loader import VideoDataset
from tools.utils import Logger
from tools.eval_metrics import evaluate, evaluate_locations, evaluate_clothes, evaluate_scales

parser = argparse.ArgumentParser(description='Test AP3D using all frames')
# Datasets
parser.add_argument('--root', type=str, default='../../mevid')
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int)
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=128)
# Augment
parser.add_argument('--test_frames', default=32, type=int, 
                    help='frames per clip for test')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='ap3dres50', 
                    help="ap3dres50, ap3dnlres50")
# Miscs
parser.add_argument('--resume', type=str, default='log-mars-ap3d', metavar='PATH')
parser.add_argument('--test_epochs', default=[240], nargs='+', type=int)
parser.add_argument('--distance', type=str, default='cosine', 
                    help="euclidean or cosine")
parser.add_argument('--gpu', default='2', type=str, 
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.resume, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)

    # Data augmentation
    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_test = None

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False)

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False)

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    for epoch in args.test_epochs:
        model_path = osp.join(args.resume, 'checkpoint_ep'+str(epoch)+'.pth.tar')
        print("Loading checkpoint from '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

        if use_gpu: model = model.cuda()

        print("Evaluate")
        with torch.no_grad():
            test(model, queryloader, galleryloader, use_gpu)


def extract111(model, vids, use_gpu):
    n, c, f, h, w = vids.size()
    assert(n == 1)

    if use_gpu:
        feat = torch.cuda.FloatTensor()
    else:
        feat = torch.FloatTensor()
    for i in range(math.ceil(f/args.test_frames)):
        clip = vids[:, :, i*args.test_frames:(i+1)*args.test_frames, :, :]
        
        if use_gpu:
            clip = clip.cuda()
        output = model(clip)
        feat = torch.cat((feat, output), 1)

    feat = feat.mean(1)
    feat = model.bn(feat)
    feat = feat.data.cpu()

    return feat


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    # test using 4 frames
    since = time.time()
    model.eval()

    qf, q_pids, q_camids, q_oids = [], [], [], []
    for batch_idx, (vids, pids, camids, oids) in enumerate(queryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)
        feat = feat.mean(1)
        feat = model.bn(feat)
        feat = feat.data.cpu()

        qf.append(feat)
        q_pids.extend(pids)
        q_camids.extend(camids)
        q_oids.extend(oids)

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    q_oids = np.asarray(q_oids)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids, g_oids = [], [], [], []
    for batch_idx, (vids, pids, camids, oids) in enumerate(galleryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)
        feat = feat.mean(1)
        feat = model.bn(feat)
        feat = feat.data.cpu()

        gf.append(feat)
        g_pids.extend(pids)
        g_camids.extend(camids)
        g_oids.extend(oids)

    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    g_oids = np.asarray(g_oids)
    #if args.dataset == 'mars':
    #    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    #    gf = torch.cat((qf, gf), 0)
    #    g_pids = np.append(q_pids, g_pids)
    #    g_camids = np.append(q_camids, g_camids)

    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m,n))

    if args.distance == 'euclidean':
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        for i in range(m):
            distmat[i:i+1].addmm_(1, -2, qf[i:i+1], gf.t())
    else:
        q_norm = torch.norm(qf, p=2, dim=1, keepdim=True)
        g_norm = torch.norm(gf, p=2, dim=1, keepdim=True)
        qf = qf.div(q_norm.expand_as(qf))
        gf = gf.div(g_norm.expand_as(gf))
        for i in range(m):
            distmat[i] = - torch.mm(qf[i:i+1], gf.t())
    distmat = distmat.numpy()


    np.savez('eval.npz', distmat=distmat, q_pids=q_pids, g_pids=g_pids, q_camids=q_camids, g_camids=g_camids, q_oids=q_oids, g_oids=g_oids)
    #data = np.load('eval.npz')
    #distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids = data['distmat'], data['q_pids'], data['g_pids'], data['q_camids'], data['g_camids'], data['q_oids'], data['g_oids']

    print("Computing CMC and mAP")
    all_cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Overall Results ----------")
    print('top1:{:.1%} top5:{:.1%} top10:{:.1%} top20:{:.1%} mAP:{:.1%}'.format(all_cmc[0],all_cmc[4],all_cmc[9],all_cmc[19],mAP))
    print("------------------")

    # attribute evaluation
    cmc, mAP = evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='SC')

    print("SC Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='CC')

    print("CC Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL')
    print("Location Results (same locations)----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")


    cmc, mAP = evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='DL')
    print("Location Results (different locations)----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS')
    print("Scale Results (same scales)----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")
    cmc, mAP = evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='DS')
    print("Scale Results (different scales)----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    return all_cmc[0]


if __name__ == '__main__':
    main()
