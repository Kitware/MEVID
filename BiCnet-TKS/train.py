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
from losses import TripletLoss, DivRegLoss
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
parser.add_argument('--max_epoch', default=150, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start_epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train_batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test_batch', default=64, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.00035, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=[40, 80, 120], nargs='+', type=int,
                    help="stepsize to decay learning rat-e")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight_decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--distance', type=str, default='consine', help="euclidean or consine")
parser.add_argument('--num_instances', type=int, default=4, help="number of instances per identity")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='BiCnet_TKS')
parser.add_argument('--save-dir', type=str, default='./results')
parser.add_argument('--resume', type=str, default='', metavar='PATH')
# Spatial Attention
parser.add_argument('--alpha', default=0.01, type=float)
# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval_step', type=int, default=40,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start_eval', type=int, default=0, help="start to evaluate after specific epoch")
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='1, 2', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
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
    spatial_transform_train = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.RandomHorizontalFlip(),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ST.RandomErasing(),
            ])

    temporal_transform_train = TT.TemporalRandomCrop(size=args.seq_len, stride=args.sample_stride)

    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    temporal_transform_test = TT.TemporalBeginCrop(size=args.test_frames)

    pin_memory = True if use_gpu else False

    dataset_train = dataset.train
    if args.dataset != 'mars':
        dataset_train = dataset.train_dense
        print('process {} dataset'.format(args.dataset))

    trainloader = DataLoader(
        VideoDataset(dataset_train, spatial_transform=spatial_transform_train, temporal_transform=temporal_transform_train),
        sampler=RandomIdentitySampler(dataset_train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids)
    #print(model)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    criterion_xent = nn.CrossEntropyLoss() 
    if args.dataset == 'lsvid':
        print('process lsvid with contrastive loss!')
        criterion_htri = ContrastiveLoss()
    else:
        print('process {} with triplet loss!'.format(args.dataset))
        criterion_htri = TripletLoss()
    criterion_div = DivRegLoss() 

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    start_epoch = args.start_epoch

    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']

    model = nn.DataParallel(model).cuda()

    start_time = time.time()
    train_time = 0
    best_mAP = -np.inf
    best_epoch = 0
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):

        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, criterion_div, optimizer, trainloader, use_gpu)
        # torch.cuda.empty_cache()
        train_time += round(time.time() - start_train_time)

        scheduler.step()

        if (epoch+1) % args.eval_step == 0 or (epoch > 100 and (epoch+1) % 10 == 0):
            print("==> Test")
            with torch.no_grad():
                mAP = test(model, queryloader, galleryloader, use_gpu)
                # torch.cuda.empty_cache()
            is_best = mAP >= best_mAP
            if is_best: 
                best_mAP = mAP
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({
                'state_dict': state_dict,
                'mAP': mAP,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best mAP {:.1%}, achieved at epoch {}".format(best_mAP, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion_xent, criterion_htri, criterion_div, optimizer, trainloader, use_gpu):
    batch_xent_loss = AverageMeter()
    batch_htri_loss = AverageMeter()
    batch_div_loss = AverageMeter()
    batch_loss = AverageMeter()
    batch_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (vids, pids, _) in enumerate(trainloader):
        if use_gpu:
            vids, pids = vids.cuda(), pids.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs, features, masks = model(vids)

        # combine hard triplet loss with cross entropy loss
        xent_loss = criterion_xent(outputs, pids)
        htri_loss = criterion_htri(features, pids)
        div_loss = criterion_div(masks)
        loss = xent_loss + htri_loss + args.alpha * div_loss

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        _, preds = torch.max(outputs.data, 1)
        batch_corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        
        batch_xent_loss.update(xent_loss.item(), pids.size(0))
        batch_htri_loss.update(htri_loss.item(), pids.size(0))
        batch_div_loss.update(div_loss.item(), pids.size(0))
        batch_loss.update(loss.item(), pids.size(0)) 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Epoch{0} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '
          'Xent:{xent.avg:.4f} '
          'Htri:{htri.avg:.4f} '
          'div_loss:{div_loss.avg:.4f} '
          'Acc:{acc.avg:.2%} '.format(
          epoch+1, batch_time=batch_time,
          data_time=data_time, loss=batch_loss,
          xent=batch_xent_loss, htri=batch_htri_loss,
          div_loss=batch_div_loss, acc=batch_corrects))


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    since = time.time()
    model.eval()

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(queryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)
        feat = feat.mean(1)
        feat = model.module.bn(feat)
        feat = feat.data.cpu()

        qf.append(feat)
        q_pids.extend(pids)
        q_camids.extend(camids)

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {} matrix".format(qf.shape))

    gf, g_pids, g_camids = [], [], []
    for batch_idx, (vids, pids, camids) in enumerate(galleryloader):
        if use_gpu:
            vids = vids.cuda()
        feat = model(vids)
        feat = feat.mean(1)
        feat = model.module.bn(feat)
        feat = feat.data.cpu()

        gf.append(feat)
        g_pids.extend(pids)
        g_camids.extend(camids)

    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    '''if args.dataset == 'mars' or args.dataset == 'lsvid':
        print('process the dataset {}!'.format(args.dataset))
        # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
        gf = torch.cat((qf, gf), 0)
        g_pids = np.append(q_pids, g_pids)
        g_camids = np.append(q_camids, g_camids)'''

    print("Extracted features for gallery set, obtained {} matrix".format(gf.shape))

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
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    return mAP

if __name__ == '__main__':
    main()
