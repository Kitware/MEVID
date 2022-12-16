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
parser.add_argument('--epochs-eval', default=[1], type=list)
parser.add_argument('--name', '--model_name', type=str, default='_bot_')
parser.add_argument('--validation-training', action='store_true', help="more useful for validation")
parser.add_argument('--resume-training', action='store_true', help="Continue training")
parser.add_argument('--gpu-devices', default='3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
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
print("initializing dataset")
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


elif args.dataset == "mot":
    print("USING MOT CONFIG")
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
    print("going through trainloader")
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

print("loading query")

queryloader = DataLoader(
    VideoDataset(dataset.query, seq_len=args.seq_len, sample='dense', transform=transform_test),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)

print("loading gallery")


galleryloader = DataLoader(
    VideoDataset(dataset.gallery, seq_len=args.seq_len, sample='dense', transform=transform_test),
    batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
    pin_memory=pin_memory, drop_last=False,
)



print("initializing model")
print(dataset.num_train_pids)
model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})
print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

criterion_htri = TripletLoss(margin, 'cosine')
criterion_xent = CrossEntropyLabelSmooth(dataset.num_train_pids)
criterion_center_loss = CenterLoss(use_gpu=use_gpu, num_classes=dataset.num_train_pids)

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


def normal_train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa=None):
    model.train()
    losses = AverageMeter()
    cetner_loss_weight = 0.0005
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        # print(batch_idx)
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
            # 32,4,3,224,112 , 32
        imgs, pids = Variable(imgs), Variable(pids)
        outputs, features = model(imgs)
        ide_loss = criterion_xent(outputs , pids)
        triplet_loss = criterion_htri(features, features, features, pids, pids, pids)
        center_loss = criterion_center_loss(features, pids)
        if args.use_OSMCAA:
            if use_gpu:
                osm_caa_loss = criterion_osm_caa(features, pids , model.module.classifier.classifier.weight.t())         
            else:
                osm_caa_loss = criterion_osm_caa(features, pids , model.classifier.classifier.weight.t()) 
            loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss
        elif args.cl_centers : 
            osm_caa_loss = criterion_osm_caa(features, pids , criterion_center_loss.centers.t()) 
            loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss
        else:
            loss = ide_loss + triplet_loss  + center_loss * cetner_loss_weight
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()
        for param in criterion_center_loss.parameters():
            param.grad.data *= (1./cetner_loss_weight)
        optimizer_center.step()
        losses.update(loss.data.item(), pids.size(0))
        if (batch_idx+1) % args.print_freq == 0:
            if args.use_OSMCAA or args.cl_centers:
                print("Batch {}/{}\t TripletLoss  ({:.6f}) OSM Loss: ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(batch_idx+1, len(trainloader) ,triplet_loss.item(), osm_caa_loss.item() ,losses.val, losses.avg))        
            else:
                print("Batch {}/{}\t TripletLoss  ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(batch_idx+1, len(trainloader) ,triplet_loss.item(),losses.val, losses.avg))        
        

def train_attn(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa=None):
    model.train()
    losses = AverageMeter()
    cetner_loss_weight = 0.0005
    for batch_idx, (imgs, pids, _, labels) in enumerate(trainloader):
        # print(batch_idx)
        if use_gpu:
            imgs, pids  , labels = imgs.cuda(), pids.cuda() , labels.cuda().float()
            # 32,4,3,224,112 , 32
        imgs, pids = Variable(imgs), Variable(pids)
        outputs, features , a_vals = model(imgs)
        ide_loss = criterion_xent(outputs , pids)
        triplet_loss = criterion_htri(features, features, features, pids, pids, pids)
        center_loss = criterion_center_loss(features, pids)
        attn_noise  = a_vals * labels
        attn_loss = attn_noise.sum(1).mean()
        if args.use_OSMCAA:
            if use_gpu:
                osm_caa_loss = criterion_osm_caa(features, pids , model.module.classifier.classifier.weight.t())         
            else:
                osm_caa_loss = criterion_osm_caa(features, pids , model.classifier.classifier.weight.t()) 
            loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss + attn_loss
        elif args.cl_centers : 
            osm_caa_loss = criterion_osm_caa(features, pids , criterion_center_loss.centers.t()) 
            loss = ide_loss + (1-beta_ratio )* triplet_loss  + center_loss * cetner_loss_weight + beta_ratio * osm_caa_loss + attn_loss
        else:
            loss = ide_loss + triplet_loss  + center_loss * cetner_loss_weight + attn_loss
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer.step()
        for param in criterion_center_loss.parameters():
            param.grad.data *= (1./cetner_loss_weight)
        optimizer_center.step()
        losses.update(loss.data.item(), pids.size(0))
        if (batch_idx+1) % args.print_freq == 0:
            if args.use_OSMCAA or args.cl_centers:
                print("Batch {}/{}\t Attn Loss  ({:.6f}) TripletLoss  ({:.6f}) OSM Loss: ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(batch_idx+1, len(trainloader), attn_loss.item() ,triplet_loss.item(), osm_caa_loss.item() ,losses.val, losses.avg))        
            else:
                print("Batch {}/{}\t Attn Loss  ({:.6f}) TripletLoss  ({:.6f}) Total Loss {:.6f} ({:.6f})  ".format(batch_idx+1, len(trainloader),attn_loss.item() ,triplet_loss.item(),losses.val, losses.avg))        
        
                
        


from tools.eval_metrics import evaluate , re_ranking
def test_rerank(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()
    query = torch.load(osp.join(args.save_dir, 'query_feat.pth'))
    qf = query['qf']
    q_pids = query["q_pids"]
    q_camids = query["q_camids"]

    gallery1 = torch.load(osp.join(args.save_dir, 'gallery_feat_1.pth'))
    gallery2 = torch.load(osp.join(args.save_dir, 'gallery_feat_2.pth'))
    gallery3 = torch.load(osp.join(args.save_dir, 'gallery_feat_3.pth'))

    gallery = {}
    for key in gallery1.keys():
        print(key)
        if key == "qf":
            tensor = torch.cat((gallery1[key], gallery2[key], gallery3[key]), dim=0)
            gallery[key] = tensor
        else:
            value = np.concatenate((gallery1[key], gallery2[key], gallery3[key]), axis=None)
            gallery[key] = value    
    
    gf = gallery['qf']
    g_pids = gallery["q_pids"]
    g_camids = gallery["q_camids"]

    print("Computing distance matrix")
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





if args.pretrained_model != '':
    name = args.pretrained_model
    # name = "ResNet50ta_bt2_supervised_erase_59_checkpoint_ep81.pth.tar"
    print("loading .... " , name)
    checkpoint = torch.load(name)
    state_dict = {}
    if args.dataset != "mars":
        for key in checkpoint['state_dict']:
            if "classifier" not in  key:
                state_dict["module." + key] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict,  strict=False)
    else:
        for key in checkpoint['state_dict']:
            state_dict["module." + key] = checkpoint['state_dict'][key]
        model.load_state_dict(state_dict,  strict=True)


args.save_dir = "/home/local/KHQ/alexander.lynch/DIY-AI/person_reid/Video-Person-Re-ID-Fantastic-Techniques-and-Where-to-Find-Them/models/log/"
args.arch += args.name +"_" + str(args.opt) + "_"
is_best = 0
prev_best = 0 

if type(args.epochs_eval[0]) != int :
    args.epochs_eval = map(int,  "".join([vals for vals in args.epochs_eval if vals != '[' and vals != ']']).split(",") )

print (args.arch)


if args.attn_loss:
    train = train_attn
else:
    train = normal_train

if not args.validation_training:
    
    print(args.epochs_eval)
    if not args.evaluate :
        print("NOT EVALLING")
        for epoch in range(0, args.max_epoch):
            print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
            train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa )
            if args.stepsize > 0 : scheduler.step()
            if epoch in args.epochs_eval:
                #re_rank_mAP = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)
                #if re_rank_mAP > prev_best:
                #prev_best = re_rank_mAP
                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                
                save_checkpoint({
                    'state_dict': state_dict,
                    }, is_best, osp.join(args.save_dir, args.arch + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))                
                print("checkpoint saved", osp.join(args.save_dir, args.arch + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    else:
        print("EVALLING")
        for epoch in args.epochs_eval:
            checkpoint = torch.load(osp.join(args.save_dir, 'ckpt_best.tar'))
            state_dict = {}
            for key in checkpoint['state_dict']:
                    state_dict["module." + key] = checkpoint['state_dict'][key]
            model.load_state_dict(state_dict,  strict=True)
            rank1 = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)

else:
    print("evaluation at every 10 epochs, Highly GPU/CPU expensive process, avoid running anything in Parallel")
    factor = 10 
    print(args.epochs_eval)
    print("====")
    print(args.start_epoch)
    for epoch in range(args.start_epoch, args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu , optimizer_center , criterion_center_loss , criterion_osm_caa )
        if args.stepsize > 0 : scheduler.step()
        if epoch in args.epochs_eval:
            re_rank_mAP = test_rerank(model, queryloader, galleryloader, args.pool, use_gpu)
            if re_rank_mAP > prev_best:
                prev_best = re_rank_mAP
                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                    save_checkpoint({
                        'state_dict': state_dict,
                        }, is_best, osp.join(args.save_dir, args.arch + '_checkpoint_ep' + str(epoch+1) + '.pth.tar'))                



# python config_trainer.py --focus=map --dataset=mars --opt=$chikka --name=_mars_attncl_centers_ --cl-centers --attn-loss >>  output/mars_attn_cl_centers_$chikka.out
# python3 config_trainer.py --focus=map --dataset=mars --opt= --name=_mars_osm_ --use-OSMCAA >>  output/mars_osm_.out


