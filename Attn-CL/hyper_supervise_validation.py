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
# from tools.transforms2 import RandomErasing
from tools.scheduler import WarmupMultiStepLR
from tools.utils import AverageMeter, Logger, save_checkpoint
from tools.samplers import RandomIdentitySampler
from tools.video_loader import VideoDataset 
import tools.data_manager as data_manager
from tools.eval_metrics import evaluate , re_ranking

from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
import ax
from typing import Dict, List, Tuple



parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
parser.add_argument('-d', '--dataset', type=str, default='mars_subset2',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max-epoch', default=800, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=200, type=int,
                    help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50tp', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
parser.add_argument('--print-freq', type=int, default=40, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--epochs-eval', default=[99,199,299,399,499,599,699,799], type=list)
parser.add_argument('--gpu-devices', default='0,1,2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-f', '--focus', type=str, default='map', help="map,rerank_map")
parser.add_argument('-s', '--sampling', type=str, default='random', help="random,intille")

args = parser.parse_args()


def train_model(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss, criterion_osm_caa, beta_ratio):
    model.train()
    losses = AverageMeter()
    cetner_loss_weight = 0.0005
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        if use_gpu:
            imgs, pids   = imgs.cuda(), pids.cuda() 
        imgs, pids = Variable(imgs), Variable(pids)
        outputs, features  = model(imgs)
        ide_loss = criterion_xent(outputs , pids)
        triplet_loss = criterion_htri(features, features, features, pids, pids, pids)
        center_loss = criterion_center_loss(features, pids)
        # hosm_loss = criterion_osm_caa(features, pids , model.module.classifier.classifier.weight.t()) 
        hosm_loss = criterion_osm_caa(features, pids , criterion_center_loss.centers.t() ) 
        
        loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * hosm_loss 
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()
        for param in criterion_center_loss.parameters():
            param.grad.data *= (1./cetner_loss_weight)
        optimizer_center.step()
        losses.update(loss.data.item(), pids.size(0))
    return (losses.avg , ide_loss.item() , triplet_loss.item() , hosm_loss.item())
    

def train(parameters: Dict[str, float]) -> nn.Module:
    global args 
    print("====", args.focus,  "=====")
    torch.manual_seed(args.seed)
    # args.gpu_devices = "0,1"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
    
    dataset = data_manager.init_dataset(name=args.dataset, sampling= args.sampling)
    transform_test = transforms.Compose([
    transforms.Resize((args.height, args.width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    pin_memory = True if use_gpu else False
    transform_train = transforms.Compose([
                transforms.Resize((args.height, args.width), interpolation=3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Pad(10),
                Random2DTranslation(args.height, args.width),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    batch_size = int(round(parameters.get("batch_size", 32) )) 
    base_learning_rate = 0.00035
    # weight_decay = 0.0005
    alpha = parameters.get("alpha", 1.2)
    sigma = parameters.get("sigma", 0.8)
    l = parameters.get("l", 0.5)
    beta_ratio = parameters.get("beta_ratio", 0.5)
    gamma = parameters.get("gamma", 0.1)
    margin = parameters.get("margin", 0.3)
    weight_decay = parameters.get("weight_decay", 0.0005)
    lamb = 0.3 
    
    num_instances = 4
    pin_memory = True
    trainloader = DataLoader(
    VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
    sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
    batch_size=batch_size, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=True,
    )

    if args.dataset == 'mars_subset' :
        validation_loader = DataLoader(
            VideoDataset(dataset.val, seq_len=8, sample='random', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )
    else:
        queryloader = DataLoader(
            VideoDataset(dataset.val_query, seq_len=args.seq_len, sample='dense_subset', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )
        galleryloader = DataLoader(
            VideoDataset(dataset.val_gallery, seq_len=args.seq_len, sample='dense_subset', transform=transform_test),
            batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
            pin_memory=pin_memory, drop_last=False,
        )

    criterion_htri = TripletLoss(margin, 'cosine')
    criterion_xent = CrossEntropyLabelSmooth(dataset.num_train_pids)
    criterion_center_loss = CenterLoss(use_gpu=1)
    criterion_osm_caa = OSM_CAA_Loss(alpha=alpha , l=l , osm_sigma=sigma )
    args.arch = "ResNet50ta_bt"
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
    if use_gpu:
        model = nn.DataParallel(model).cuda()
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = base_learning_rate
        weight_decay = weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.Adam(params)
    scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70], gamma=gamma, warmup_factor=0.01, warmup_iters=10)
    optimizer_center = torch.optim.SGD(criterion_center_loss.parameters(), lr=0.5)
    start_epoch = args.start_epoch
    best_rank1 = -np.inf
    num_epochs = 121
    
    if 'mars' not in args.dataset :
        num_epochs = 121
    # test_rerank(model, queryloader, galleryloader, args.pool, use_gpu, lamb=lamb , parameters=parameters)
    for epoch in range (num_epochs):
        vals = train_model(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss, criterion_osm_caa, beta_ratio)
        if math.isnan(vals[0]):
            return 0
        scheduler.step()
        if epoch % 40 ==0 :
            print("TripletLoss {:.6f} OSM Loss {:.6f} Cross_entropy {:.6f} Total Loss {:.6f}  ".format(vals[1] , vals[3] , vals[1] , vals[0]))            
    
    if args.dataset == 'mars_subset' :
        result1 = test_validation(model, validation_loader, args.pool, use_gpu,  parameters=parameters)
        del validation_loader
    else:
        result1= test_rerank(model, queryloader, galleryloader, args.pool, use_gpu, lamb=lamb , parameters=parameters)    
        del queryloader
        del galleryloader
    del trainloader 
    del model
    del criterion_htri
    del criterion_xent
    del criterion_center_loss
    del criterion_osm_caa
    del optimizer
    del optimizer_center
    del scheduler
    return result1





def test_validation(model, validation_loader, pool, use_gpu, ranks=[1, 5, 10, 20], lamb=0.3, parameters=None):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(validation_loader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, s, c, h, w = imgs.size()

        assert(b==1)
        n = b * s // 4
        imgs = imgs.view(n , 4, c, h, w)
        features = model(imgs)
        features = features.view(n, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    gf, g_pids, g_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(validation_loader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, s, c, h, w = imgs.size()
        n = b*s//4
        imgs = imgs.view(n, 4 , c, h, w)
        assert(b==1)
        features = model(imgs)
        features = features.view(n, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    distmat_rerank = re_ranking(qf,gf , lambda_value=lamb)
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Rerank Computing CMC and mAP")
    re_rank_cmc, re_rank_mAP = evaluate(distmat_rerank, q_pids, g_pids, q_camids, g_camids)
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print (parameters)
    print("Results ---------- ")
    if 'mars' in args.dataset :
        print("mAP: {:.1%} vs {:.1%}".format(mAP, re_rank_mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%} vs {:.1%}".format(r, cmc[r-1], re_rank_cmc[r-1]))
    print("------------------")
    del qf, q_pids, q_camids
    del gf, g_pids, g_camids
    del distmat , distmat_rerank
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




def test_rerank(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20], lamb=0.3, parameters=None):
    model.eval()
    qf, q_pids, q_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, n, s, c, h, w = imgs.size()
        assert(b==1)
        imgs = imgs.view(b*n, s, c, h, w)
        features = model(imgs)
        features = features.view(n, -1)
        features = torch.mean(features, 0)
        features = features.data.cpu()
        qf.append(features)
        q_pids.extend(pids)
        q_camids.extend(camids)
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
    gf, g_pids, g_camids = [], [], []
    for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            imgs = imgs.cuda()
        imgs = Variable(imgs, volatile=True)
        b, n, s, c, h, w = imgs.size()
        imgs = imgs.view(b*n, s , c, h, w)
        assert(b==1)
        features = model(imgs)
        features = features.view(n, -1)
        if pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)
        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)
    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    gf = gf.numpy()
    qf = qf.numpy()
    distmat_rerank = re_ranking(qf,gf , lambda_value=lamb)
    print("Original Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    print("Rerank Computing CMC and mAP")
    re_rank_cmc, re_rank_mAP = evaluate(distmat_rerank, q_pids, g_pids, q_camids, g_camids)
    # print("Results ---------- {:.1%} ".format(distmat_rerank))
    print (parameters)
    print("Results ---------- ")
    if 'mars' in args.dataset :
        print("mAP: {:.1%} vs {:.1%}".format(mAP, re_rank_mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%} vs {:.1%}".format(r, cmc[r-1], re_rank_cmc[r-1]))
    print("------------------")
    del qf, q_pids, q_camids
    del gf, g_pids, g_camids
    del distmat , distmat_rerank
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


best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "sigma", "type": "range", "bounds": [1e-1, 1.0]},
        {"name": "alpha", "type": "range", "bounds": [0.5, 3.0]},
        {"name": "l", "type": "range", "bounds": [1e-1, 1.0]},
        {"name": "margin", "type": "range", "bounds": [1e-6, 1.0], "log_scale": True},
        {"name": "beta_ratio", "type": "range", "bounds": [1e-6, 1.0]},
        {"name": "gamma", "type": "range", "bounds": [1e-6, 1.0]},
        {"name": "weight_decay", "type": "range", "bounds": [1e-6, 1.0]},
        # {"name": "batch_size", "type": "range", "bounds": [10, 80]},
        ],
    evaluation_function=train,
    objective_name='ranking',
    minimize=False,
    total_trials = 60,
)

print("===========")
print(best_parameters)
print("===========")
print(values)

