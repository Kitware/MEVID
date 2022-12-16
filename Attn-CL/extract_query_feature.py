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
parser.add_argument('-a', '--arch', type=str, default='ResNet50ta_bt', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])


# Miscs
parser.add_argument('--print-freq', type=int, default=40, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--pretrained-model', type=str, default='log/ResNet50ta_bt_CL_CENTERS__checkpoint_ep541.pth.tar', help='need to be set for resnet3d models')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--save-dir', type=str, default='log/')
parser.add_argument('--epochs-eval', default=[20 * i for i in range(1,80)], type=list)
parser.add_argument('--name', '--model_name', type=str, default='_bot_')
parser.add_argument('--validation-training', action='store_true', help="more useful for validation")
parser.add_argument('--resume-training', action='store_true', help="Continue training")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('-f', '--focus', type=str, default='map', help="map,rerank_map")
parser.add_argument('-opt', '--opt', type=str, default='3', help="choose opt")
parser.add_argument('-s', '--sampling', type=str, default='random', help="choose sampling for training")

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
    

if args.dataset == "mars" or args.dataset == "mot":
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
#print(state_dict['state_dict'].keys())
model = models.init_model(name=args.arch, num_classes=625, loss={'xent', 'htri'})
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
qf, q_pids, q_camids = [], [], []
for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
    if use_gpu:
        imgs = imgs.cuda()
    with torch.no_grad():
        imgs = Variable(imgs)
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

query = {
        'qf': qf,
        'q_pids': q_pids,
        'q_camids': q_camids,
    }
torch.save(query, osp.join(args.save_dir, 'query_feat.pth'))
print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
