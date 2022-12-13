from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

from torch.utils.data import DataLoader

import data_manager
from samplers import RandomIdentitySampler
from video_loader import VideoDataset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from lr_schedulers import WarmupMultiStepLR
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, TripletLoss
from utils import AverageMeter, Logger, make_optimizer, DeepSupervision
from eval_metrics import evaluate_reranking
from config import cfg

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument("--config_file", default="./configs/softmax_triplet.yml", help="path to config file", type=str)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
parser.add_argument('--arch', type=str, default='ResNet50', choices=['ResNet50', 'PSTA'])
parser.add_argument('--train_sampler', type=str, default='Random_interval', help='train sampler', choices=['Random_interval','Random_choice'])
parser.add_argument('--test_sampler', type=str, default='Begin_interval', help='test sampler', choices=['dense', 'Begin_interval'])
parser.add_argument('--triplet_distance', type=str, default='cosine', choices=['cosine','euclidean'])
parser.add_argument('--test_distance', type=str, default='cosine', choices=['cosine','euclidean'])
parser.add_argument('--split_id', type=int, default=0)
parser.add_argument('--dataset', type=str, default='mars', choices=['mot', 'mars','duke'])
parser.add_argument('--seq_len', type=int, default=8)


args_ = parser.parse_args()

if args_.config_file != "":
    cfg.merge_from_file(args_.config_file)
cfg.merge_from_list(args_.opts)

tqdm_enable = False

def main():
    #runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, runId)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    print(cfg.OUTPUT_DIR)
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    use_gpu = torch.cuda.is_available() and cfg.MODEL.DEVICE == "cuda"
    sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, 'log_train_gt.txt'))

    print("=========================\nConfigs:{}\n=========================".format(cfg))
    s = str(args_).split(", ")
    print("Fine-tuning detail:")
    for i in range(len(s)):
        print(s[i])
    print("=========================")

    if use_gpu:
        print("Currently using GPU {}".format(cfg.MODEL.DEVICE_ID))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(cfg.DATASETS.NAME))


    dataset = data_manager.init_dataset(root=cfg.DATASETS.ROOT_DIR, name=args_.dataset, split_id = args_.split_id)
    print("Initializing model: {}".format(cfg.MODEL.NAME))

    model = models.init_model(name=args_.arch, num_classes=dataset.num_train_pids, pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
                              model_name=cfg.MODEL.NAME, seq_len = args_.seq_len)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    transform_train = T.Compose([
        T.resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
 #       T.random_crop((256,128)),
 #       T.pad(10),
        T.random_horizontal_flip(),
        T.to_tensor(),
        T.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.random_erasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    transform_test = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pin_memory = True if use_gpu else False
    video_sampler = RandomIdentitySampler(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE)

    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args_.seq_len, sample=args_.train_sampler, transform=transform_train,
                     dataset_name=args_.dataset),
        sampler=video_sampler,
        batch_size=cfg.SOLVER.SEQS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=True
    )

    if args_.test_sampler == 'dense':
        print('Build dense sampler')
        queryloader = DataLoader(
            VideoDataset(dataset.query, seq_len=args_.seq_len, sample=args_.test_sampler, transform=transform_test,
                         max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
            batch_size=1 , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=False
        )

        galleryloader = DataLoader(
            VideoDataset(dataset.gallery, seq_len=args_.seq_len, sample=args_.test_sampler, transform=transform_test,
                         max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
            batch_size=1 , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=False,
        )
    else:
        queryloader = DataLoader(
            VideoDataset(dataset.query, seq_len=args_.seq_len, sample=args_.test_sampler,
                         transform=transform_test,
                         max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
            batch_size=cfg.TEST.SEQS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=False
        )

        galleryloader = DataLoader(
            VideoDataset(dataset.gallery, seq_len=args_.seq_len, sample=args_.test_sampler,
                         transform=transform_test,
                         max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME),
            batch_size=cfg.TEST.SEQS_PER_BATCH, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=pin_memory, drop_last=False,
        )

    model = nn.DataParallel(model)
    model.cuda()

    start_time = time.time()
    xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids)

    tent = TripletLoss(cfg.SOLVER.MARGIN, distance=args_.triplet_distance)

    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    start_epoch = 0
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):

        print("==> Epoch {}/{}".format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
        print("current lr:", scheduler.get_lr()[0])

        train(model, trainloader, xent, tent, optimizer, use_gpu)
        scheduler.step()
        torch.cuda.empty_cache()

        if cfg.SOLVER.EVAL_PERIOD > 0 and ((epoch + 1) % cfg.SOLVER.EVAL_PERIOD == 0 or (epoch + 1) == cfg.SOLVER.MAX_EPOCHS) or epoch == 0:
            print("==> Test")
            _, metrics = test(model, queryloader, galleryloader, use_gpu)
            rank1 = metrics[0]
            if epoch>220:
                state_dict = model.state_dict()
                torch.save(state_dict, osp.join(cfg.OUTPUT_DIR, 'checkpoint_ep' + str(epoch + 1) + '.pth'))


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, trainloader, xent, tent, optimizer, use_gpu):

    model.train()
    xent_losses = AverageMeter()
    tent_losses = AverageMeter()
    losses = AverageMeter()

    for batch_idx, (imgs, pids, _, _) in enumerate(trainloader):

        optimizer.zero_grad()
        if use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
        outputs, features = model(imgs)

        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(xent, outputs, pids)
        else:
            xent_loss = xent(outputs, pids)

        if isinstance(features, (tuple, list)):
            tent_loss = DeepSupervision(tent, features, pids)
        else:
            tent_loss = tent(features, pids)

        xent_losses.update(xent_loss.item(), 1)
        tent_losses.update(tent_loss.item(), 1)

        loss = xent_loss + tent_loss
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), 1)

    print("Batch {}/{}\t Loss {:.6f} ({:.6f}) xent Loss {:.6f} ({:.6f}), tent Loss {:.6f} ({:.6f})".format(
        batch_idx + 1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, tent_losses.val, tent_losses.avg))
    return losses.avg


def test(model, queryloader, galleryloader, use_gpu, ranks=[1,5,10,20]):

    with torch.no_grad():
        model.eval()
        qf, q_pids, q_camids =  [], [], []
        query_pathes = []
        for batch_idx, (imgs, pids, camids, img_path) in enumerate(tqdm(queryloader)):
            query_pathes.append(img_path[0])
            del img_path
            if use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                camids = camids.cuda()

            if len(imgs.size()) == 6:
                method = 'dense'
                b, n, s, c, h, w = imgs.size()
                assert (b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
            else:
                method = None

            features, pids, camids = model(imgs, pids, camids)
            q_pids.extend(pids.data.cpu())
            q_camids.extend(camids.data.cpu())
            
            features = features.data.cpu()
            torch.cuda.empty_cache()
            features = features.view(-1, features.size(1))

            if method == 'dense':
                features = torch.mean(features, 0,keepdim=True)
            qf.append(features)

        qf = torch.cat(qf,0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        np.save("query_pathes", query_pathes)


        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        gallery_pathes = []
        for batch_idx, (imgs, pids, camids, img_path) in enumerate(tqdm(galleryloader)):
            gallery_pathes.append(img_path[0])
            if use_gpu:
                imgs = imgs.cuda()
                pids = pids.cuda()
                camids = camids.cuda()

            if len(imgs.size()) == 6:
                method = 'dense'
                b, n, s, c, h, w = imgs.size()
                assert (b == 1)
                imgs = imgs.view(b * n, s, c, h, w)
            else:
                method = None

            features, pids, camids = model(imgs, pids, camids)
            features = features.data.cpu()
            torch.cuda.empty_cache()
            features = features.view(-1, features.size(1))

            if method == 'dense':
                features = torch.mean(features, 0, keepdim=True)

            g_pids.extend(pids.data.cpu())
            g_camids.extend(camids.data.cpu())
            gf.append(features)

        gf = torch.cat(gf,0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        '''if args_.dataset == 'mars':
            # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
            gf = torch.cat((qf, gf), 0)
            g_pids = np.append(q_pids, g_pids)
            g_camids = np.append(q_camids, g_camids)'''

        np.save("gallery_pathes", gallery_pathes)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")

        be_cmc, metrics = evaluate_reranking(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, args_.test_distance)
        return metrics, be_cmc

if __name__ == '__main__':

    main()




