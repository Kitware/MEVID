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
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=224,
                    help="height of an image (default: 224)")
parser.add_argument('--width', type=int, default=112,
                    help="width of an image (default: 112)")
parser.add_argument('--seq-len', type=int, default=4, help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max-epoch', default=400, type=int,
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
parser.add_argument('--use-OSMCAA', action='store_true', default=False,
                    help="Use OSM CAA loss in addition to triplet")
parser.add_argument('--cl-centers', action='store_true', default=False,
                    help="Use cl centers verison of OSM CAA loss")
parser.add_argument('--attn-loss', action='store_true', default=False,
                    help="Use attention loss")

# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50tp', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])


# Miscs
parser.add_argument('--print-freq', type=int, default=40, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='', help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--epochs-eval', default=[20 * i for i in range(1,80)], type=list)
parser.add_argument('--name', '--model_name', type=str, default='_bot_')
parser.add_argument('--validation-training', action='store_true', help="more useful for validation")
parser.add_argument('--resume-training', action='store_true', help="Continue training")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-f', '--focus', type=str, default='map', help="map,rerank_map")
parser.add_argument('-opt', '--opt', type=str, default='3', help="choose opt")
parser.add_argument('-s', '--sampling', type=str, default='random', help="choose sampling for training")

parser.add_argument('--start-index', type=int, default=0, help="starting index for gallery loader iteration")
parser.add_argument('--end-index', type=int, default=10000, help="ending index for gallery loader iteration")
parser.add_argument('--save-name', type=str, default="gallery_feat.pth")

args = parser.parse_args()

torch.manual_seed(args.seed)

# args.gpu_devices = "0,1"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
use_gpu = torch.cuda.is_available()

cudnn.benchmark = True
torch.cuda.manual_seed_all(args.seed)

dataset = data_manager.init_dataset(name=args.dataset)
pin_memory = True if use_gpu else False

transform_train = transforms.Compose([
            transforms.Resize((args.height, args.width), interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            Random2DTranslation(args.height, args.width),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing2(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])

transform_test = transforms.Compose([
    transforms.Resize((args.height, args.width), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




import configparser
config = configparser.ConfigParser()

dirpath = os.getcwd() 
opt = args.opt
    

if args.dataset == "mars":
    print("USING MARS CONFIG")
    if args.attn_loss:
        print("val.conf" , "========== ", opt ,"===============")
        config.read(dirpath + "/tools/val.conf") 
    else:
        print("val.conf" , "========== ", opt ,"===============")
        config.read(dirpath + "/tools/val.conf") 
        # print("cl_centers.conf" , "========== ", opt ,"===============")
        # config.read(dirpath + "/tools/cl_centers.conf")        
    
    sigma = float(config[opt]['sigma'])
    alpha =  float(config[opt]['alpha'])
    l = float(config[opt]['l'])
    margin =  float(config[opt]['margin'])
    beta_ratio = float(config[opt]['beta_ratio'])
    gamma  = float(config[opt]['gamma'])
    weight_decay = float(config[opt]['weight_decay'])
    if 'batch_size' in config[opt]:
        batch_size = int(config[opt]['batch_size'])
    else:
        batch_size = 32


elif args.dataset == "prid":
    print("USING PRID CONFIG")
    print("prid.conf" , "========== ", opt ,"===============")
    config.read(dirpath + "/tools/prid.conf")        
    sigma = float(config[opt]['sigma'])
    alpha =  float(config[opt]['alpha'])
    l = float(config[opt]['l'])
    margin =  float(config[opt]['margin'])
    beta_ratio = float(config[opt]['beta_ratio'])
    gamma  = float(config[opt]['gamma'])
    weight_decay = float(config[opt]['weight_decay'])
    if 'batch_size' in config[opt]:
        batch_size = int(config[opt]['batch_size'])
    else:
        batch_size = 32


if args.attn_loss:
    trainloader = DataLoader(
        VideoDataset_inderase(dataset.train, seq_len=args.seq_len, sample=args.sampling,transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )
    args.arch = "ResNet50ta_bt2"
else:
    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args.seq_len, sample='random',transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )
    args.arch = "ResNet50ta_bt"

queryloader = DataLoader(
    VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)


galleryloader = DataLoader(
    VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)



state_dict = torch.load(args.pretrained_model)
model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
model.load_state_dict(torch.load(args.pretrained_model)['state_dict'])
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

criterion_htri = TripletLoss(margin, 'cosine')
criterion_xent = CrossEntropyLabelSmooth(dataset.num_train_pids)
criterion_center_loss = CenterLoss(use_gpu=use_gpu)

if args.use_OSMCAA:
    print("USING OSM LOSS")
    print ("config, alpha = %f  sigma = %f  l=%f"%(alpha, sigma, l )  )
    criterion_osm_caa = OSM_CAA_Loss(alpha=alpha , l=l , osm_sigma=sigma )
else:
    criterion_osm_caa = None


if args.cl_centers:
    print("USING CL CENTERS")
    print ("config, alpha = %f  sigma = %f  l=%f"%(alpha, sigma, l )  )
    criterion_osm_caa = OSM_CAA_Loss(alpha=alpha , l=l , osm_sigma=sigma )


base_learning_rate =  0.00035
params = []
for key, value in model.named_parameters():
    if not value.requires_grad:
        continue
    lr = base_learning_rate
    params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]


optimizer = torch.optim.Adam(params)
scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70], gamma=gamma, warmup_factor=0.01, warmup_iters=10)
optimizer_center = torch.optim.SGD(criterion_center_loss.parameters(), lr=0.5)

start_epoch = args.start_epoch
if use_gpu:
    model = nn.DataParallel(model).cuda()


best_rank1 = -np.inf
model.eval()
print("evaluating")
gf, g_pids, g_camids = [], [], []
for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
    if batch_idx < args.start_index:
        continue

    if batch_idx % 100 == 0:
        print(batch_idx, "/", len(galleryloader))
    if use_gpu:
        imgs = imgs.cuda()
    with torch.no_grad():
        imgs = Variable(imgs)
        b, n, s, c, h, w = imgs.size()
        imgs = imgs.view(b*n, s , c, h, w)
        assert(b==1)
        features = model(imgs)
        features = features.view(n, -1)

        if args.pool == 'avg':
            features = torch.mean(features, 0)
        else:
            features, _ = torch.max(features, 0)

        features = features.data.cpu()
        gf.append(features)
        g_pids.extend(pids)
        g_camids.extend(camids)

    if batch_idx > args.end_index:
        break
gf = torch.stack(gf)
g_pids = np.asarray(g_pids)
g_camids = np.asarray(g_camids)

gallery = {
        'gf': gf,
        'g_pids': g_pids,
        'g_camids': g_camids,
    }
torch.save(gallery, osp.join(args.save_dir, args.save_name))
print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
print(args.save_dir)