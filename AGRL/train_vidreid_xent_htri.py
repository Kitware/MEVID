from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import pdb
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from torchreid import data_manager, metrics, lr_scheduler
from torchreid.dataset_loader import VideoDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, TripletLoss, DeepSupervision
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import set_wd, cur_time
from torchreid.utils.reidtools import visualize_ranked_results, calc_splits
from torchreid.utils.re_ranking import re_ranking
from torchreid.utils.model_complexity import compute_model_complexity
from torchreid.samplers import *
from torchreid.optimizers import init_optim

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='../../mevid/',
                    help="root path to data directory")
parser.add_argument('-d', '--dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=8, type=int,
                    help="number of data loading workers, suggest 4 * num_gpu (default: 8)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--seq-len', type=int, default=15,
                    help="number of images to sample in a tracklet")
parser.add_argument('--split-id', type=int, default=0,
                    help="split index (0-based), for prid2011 and ilidsvid")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=5, type=int,
                    help="test batch size (number of tracklets)")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--train-sample', default='restricted', choices=['evenly', 'random', 'consecutive', 'restricted'],
                    help="sampling strategy in training stage.")
parser.add_argument('--test-sample', default='dense', choices=['evenly', 'all', 'dense', 'skipdense'],
                    help="sampling strategy in testing stage. ")
parser.add_argument('--train-sampler', default='RandomIdentitySampler',
                    help='sampler used in training.')
# Optimization options
parser.add_argument('--optim', type=str, default='adam',
                    help="optimization algorithm (see optimizers.py)")
# Loss function
parser.add_argument('--margin', type=float, default=0.3,
                    help="margin for triplet loss")
parser.add_argument('--soft-margin', action='store_true',
                    help="soft margin for triplet loss")
parser.add_argument('--lambda-xent', type=float, default=1,
                    help="weight to balance cross entropy loss")
parser.add_argument('--lambda-htri', type=float, default=1,
                    help="weight to balance hard triplet loss")
parser.add_argument('--label-smooth', action='store_true',
                    help="use label smoothing regularizer in cross entropy loss")
# LR scheduler
parser.add_argument('--max-epoch', default=600, type=int,
                    help="maximum epochs to run")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate")
parser.add_argument('--stepsize', default=[200, 400], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--zero-wd', type=int, default=-1,
                    help='set weight decay to zero at which epoch.')
parser.add_argument('--warmup', action='store_true',
                    help='enable warmup lr scheduler.')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])
parser.add_argument('--last-stride', type=int, default=1, choices=[1, 2],
                    help='last stride in resnet50')
parser.add_argument('--num-split', type=int, default=4,
                    help='number of splits for horizontally spliting.')
parser.add_argument('--num-parts', type=int, default=3,
                    help='number of human parts, e.g. head, trunk, leg (default: 3).')
parser.add_argument('--num-gb', type=int, default=2,
                    help='number of graph block.')
parser.add_argument('--num-scale', type=int, default=1,
                    help='number of scales, used for extract multi-scale features.')
parser.add_argument('--pyramid-part', action='store_true',
                    help='enable pyramid part construction')
parser.add_argument('--use-pose', action='store_true',
                    help='use pose for graph representation learning.')
parser.add_argument('--learn-graph', action='store_true',
                    help='learn the graph.')
parser.add_argument('--knn', default=16, type=int,
                    help='k nearest neighbor in gnn layer')
parser.add_argument('--consistent-loss', action='store_true',
                    help='use sub frames to calculate consistent loss.')
parser.add_argument('--bnneck', action='store_true',
                    help='enable bn neck in the network')
# Augmentation
parser.add_argument('--flip-aug', action='store_true',
                    help='use horizontally flip for augmentation.')
parser.add_argument('--rand-erase', action='store_true',
                    help='enable random erase augmentation.')
parser.add_argument('--rand-crop', action='store_true',
                    help='enable random crop augmentation.')
parser.add_argument('--misalign-aug', action='store_true',
                    help='enable misalignment augmentation')
# Visualization
parser.add_argument('--visualize-ranks', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")
# Post Process
parser.add_argument('--dist-metric', type=str, default='euclidean',
                    help='distance metric')
parser.add_argument('--re-rank', action='store_true',
                    help='enable re-ranking in the testing stage.')
# Checkpoint
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
# Evaluation
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--eval-step', type=int, default=-1,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
# Devices
parser.add_argument('--use-cpu', action='store_true',
                    help="use cpu")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--use-avai-gpus', action='store_true',
                    help="use available gpus instead of specified devices (this is useful when using managed clusters)")
# Miscs
parser.add_argument('--print-freq', type=int, default=200,
                    help="print frequency")
parser.add_argument('--print-last', action='store_true',
                    help="print last batch in each epoch (default: False)")
parser.add_argument('--seed', type=int, default=0xff,
                    help="manual seed")
parser.add_argument('--save-dir', type=str, default='log')

# global variables
args = parser.parse_args()
best_rank1 = -np.inf
best_mAP = 0


def main():
    global best_rank1, best_mAP
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train{}.txt'.format(time.strftime('-%Y-%m-%d-%H-%M-%S'))))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test{}.txt'.format(time.strftime('-%Y-%m-%d-%H-%M-%S'))))
    writer = SummaryWriter(log_dir=args.save_dir, comment=args.arch)
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = False if 'resnet3dt' in args.arch else True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_vidreid_dataset(root=args.root, name=args.dataset, split_id=args.split_id,
                                                use_pose=args.use_pose)

    transform_train = list()
    print('Transform:')
    if args.misalign_aug:
        print('+ Misalign Augmentation')
        transform_train.append(T.GroupMisAlignAugment())
    if args.rand_crop:
        print('+ Random Crop')
        transform_train.append(T.GroupRandomCrop(size=(240, 120)))
    print('+ Resize to ({} x {})'.format(args.height, args.width))
    transform_train.append(T.GroupResize((args.height, args.width)))
    if args.flip_aug:
        print('+ Random HorizontalFlip')
        transform_train.append(T.GroupRandomHorizontalFlip())
    print('+ ToTensor')
    transform_train.append(T.GroupToTensor())
    print('+ Normalize with mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]')
    transform_train.append(T.GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if args.rand_erase:
        print('+ Random Erasing')
        transform_train.append(T.GroupRandomErasing())
    transform_train = T.Compose(transform_train)

    transform_test = T.Compose([
        T.GroupResize((args.height, args.width)),
        T.GroupToTensor(),
        T.GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=args.seq_len, sample=args.train_sample, transform=transform_train,
                     training=True, pose_info=dataset.process_poses,
                     num_split=args.num_split, num_parts=args.num_parts, num_scale=args.num_scale,
                     pyramid_part=args.pyramid_part, enable_pose=args.use_pose),
        sampler=eval(args.train_sampler)(dataset.train, batch_size=args.train_batch,
                                         num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=args.seq_len, sample=args.test_sample, transform=transform_test,
                     pose_info=dataset.process_poses, num_split=args.num_split, num_parts=args.num_parts,
                     num_scale=args.num_scale, pyramid_part=args.pyramid_part, enable_pose=args.use_pose),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=args.seq_len, sample=args.test_sample, transform=transform_test,
                     pose_info=dataset.process_poses, num_split=args.num_split, num_parts=args.num_parts,
                     num_scale=args.num_scale, pyramid_part=args.pyramid_part, enable_pose=args.use_pose),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, loss={'xent', 'htri'},
                              last_stride=args.last_stride, num_parts=args.num_parts, num_scale=args.num_scale,
                              num_split=args.num_split, pyramid_part=args.pyramid_part, num_gb=args.num_gb,
                              use_pose=args.use_pose, learn_graph=args.learn_graph, consistent_loss=args.consistent_loss,
                              bnneck=args.bnneck, save_dir=args.save_dir)

    input_size = sum(calc_splits(args.num_split)) if args.pyramid_part else args.num_split
    input_size *= args.num_scale * args.seq_len
    num_params, flops = compute_model_complexity(model,
                                                 input=[torch.randn(1, args.seq_len, 3, args.height, args.width),
                                                        torch.ones(1, input_size, input_size)],
                                                 verbose=True,
                                                 only_conv_linear=False)
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if args.label_smooth:
        criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        criterion_xent = nn.CrossEntropyLoss()
    criterion_htri = TripletLoss(margin=args.margin, soft=args.soft_margin)

    param_groups = model.parameters()
    optimizer = init_optim(args.optim, param_groups, args.lr, args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma)
    if args.warmup:
        scheduler = lr_scheduler.WarmupMultiStepLR(optimizer, milestones=args.stepsize, gamma=args.gamma,
                                                   warmup_iters=10, warmup_factor=0.01)

    if args.load_weights and check_isfile(args.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        checkpoint = torch.load(args.load_weights)
        pretrain_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(args.load_weights))

    if args.resume and check_isfile(args.resume):
        print("Loaded checkpoint from '{}'".format(args.resume))
        from functools import partial
        import pickle
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage, pickle_module=pickle)

        print('Loaded model weights')
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None and 'optimizer' in checkpoint:
            print('Loaded optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            if use_gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        start_epoch = checkpoint['epoch'] + 1
        print('- start_epoch: {}'.format(start_epoch))
        best_rank1 = checkpoint['rank1']
        print("- rank1: {}".format(best_rank1))
        if 'mAP' in checkpoint:
            best_mAP = checkpoint['mAP']
            print("- mAP: {}".format(best_mAP))
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        distmat = test(model, queryloader, galleryloader, args.pool, use_gpu, return_distmat=True)
        if args.visualize_ranks:
            visualize_ranked_results(
                distmat, dataset,
                save_dir=osp.join(args.save_dir, 'ranked_results'),
                topk=20,
            )
        return

    start_time = time.time()
    train_time = 0
    best_epoch = start_epoch
    print("==> Start training")

    for epoch in range(start_epoch, args.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, writer=writer)
        train_time += round(time.time() - start_train_time)

        if epoch >= args.zero_wd > 0:
            set_wd(optimizer, 0)
            for group in optimizer.param_groups:
                assert group['weight_decay'] == 0, '{} is not zero'.format(group['weight_decay'])

        scheduler.step(epoch)

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1, mAP = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1

            if is_best:
                best_rank1 = rank1
                best_mAP = mAP
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'rank1': rank1,
                'mAP': mAP,
                'epoch': epoch,
            }, False, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

            writer.add_scalar(tag='acc/rank1', scalar_value=rank1, global_step=epoch + 1)
            writer.add_scalar(tag='acc/mAP', scalar_value=mAP, global_step=epoch + 1)

    print("==> Best Rank-1 {:.2%}, mAP: {:.2%}, achieved at epoch {}".format(best_rank1, best_mAP, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, writer=None):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    precisions = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    end = time.time()
    for batch_idx, (imgs, pids, _, _, adj) in enumerate(trainloader):
        data_time.update(time.time() - end)
        
        if use_gpu:
            imgs, pids, adj = imgs.cuda(), pids.cuda(), adj.cuda()

        outputs, features = model(imgs, adj)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            xent_loss = criterion_xent(outputs, pids)

        if isinstance(features, tuple) or isinstance(features, list):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids)

        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        xent_losses.update(xent_loss.item(), pids.size(0))
        htri_losses.update(htri_loss.item(), pids.size(0))

        precisions.update(metrics.accuracy(outputs, pids).mean(axis=0)[0])

        if ((batch_idx + 1) % args.print_freq == 0) or (args.print_last and batch_idx == (len(trainloader) - 1)):
            num_batches = len(trainloader)
            eta_seconds = batch_time.avg * (num_batches - (batch_idx + 1) + (args.max_epoch - (epoch + 1)) * num_batches)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print('CurTime: {0}\t'
                  'Epoch: [{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {speed:.3f} samples/s\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                  'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                  'Top1 {prec.val:.4f} ({prec.avg:.4f})\t'
                  'Eta {eta}'.format(
                   cur_time(),
                   epoch + 1, batch_idx + 1, len(trainloader),
                   speed=1 / batch_time.avg * imgs.shape[0],
                   batch_time=batch_time,
                   data_time=data_time,
                   xent=xent_losses,
                   htri=htri_losses,
                   prec=precisions,
                   eta=eta_str))
        
        end = time.time()
    writer.add_scalar(tag='loss/xent_loss', scalar_value=xent_losses.avg, global_step=epoch + 1)
    writer.add_scalar(tag='loss/htri_loss', scalar_value=htri_losses.avg, global_step=epoch + 1)


def test(model, queryloader, galleryloader, pool, use_gpu, ranks=(1, 5, 10, 20), return_distmat=False):
    global mAP
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids, q_oids = [], [], [], []
        for batch_idx, (imgs, pids, camids, oids, adj) in enumerate(queryloader):
            if use_gpu:
                imgs, adj = imgs.cuda(), adj.cuda()
            if args.test_sample in ['dense', 'skipdense']:
                b, n, s, c, h, w = imgs.size()
                imgs = imgs.view(b * n, s, c, h, w)
                adj = adj.view(b * n, adj.size(-1), adj.size(-1))
            else:
                n, s, c, h, w = imgs.size()

            end = time.time()
            features = model(imgs, adj)
            batch_time.update(time.time() - end)
            if args.test_sample in ['dense', 'skipdense']:
                features = features.view(n, 1, -1)
                if pool == 'avg':
                    features = torch.mean(features, 0)
                else:
                    features, _ = torch.max(features, 0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids.numpy())
            q_camids.extend(camids.numpy())
            q_oids.extend(oids.numpy())
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        q_oids = np.asarray(q_oids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids, g_oids = [], [], [], []
        for batch_idx, (imgs, pids, camids, oids, adj) in enumerate(galleryloader):
            if use_gpu:
                imgs, adj = imgs.cuda(), adj.cuda()
            if args.test_sample in ['dense', 'skipdense']:
                b, n, s, c, h, w = imgs.size()
                imgs = imgs.view(b * n, s, c, h, w)
                adj = adj.view(b * n, adj.size(-1), adj.size(-1))
            else:
                n, s, c, h, w = imgs.size()

            end = time.time()
            features = model(imgs, adj)
            batch_time.update(time.time() - end)
            if args.test_sample in ['dense', 'skipdense']:
                features = features.view(n, 1, -1)
                if pool == 'avg':
                    features = torch.mean(features, 0)
                else:
                    features, _ = torch.max(features, 0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids.numpy())
            g_camids.extend(camids.numpy())
            g_oids.extend(oids.numpy())
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        g_oids = np.asarray(g_oids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch*args.seq_len))

    print('Computing distance matrix with metric={} ...'.format(args.dist_metric))
    distmat = metrics.compute_distance_matrix(qf, gf, args.dist_metric)
    distmat = distmat.numpy()

    if args.re_rank:
        print('Applying person re-ranking ...')
        distmat_qq = metrics.compute_distance_matrix(qf, qf, args.dist_metric)
        distmat_gg = metrics.compute_distance_matrix(gf, gf, args.dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    np.savez('eval.npz', distmat=distmat, q_pids=q_pids, g_pids=g_pids, q_camids=q_camids, g_camids=g_camids, q_oids=q_oids, g_oids=g_oids)
    
    print("Computing CMC and mAP")
    cmc, mAP = metrics.evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_mars=True)

    print("Overall Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = metrics.evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='SC')

    print("SC Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = metrics.evaluate_clothes(distmat, q_pids, g_pids, q_camids, g_camids, q_oids, g_oids, mode='CC')

    print("CC Results ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = metrics.evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='SL')
    print("Location Results (same locations)----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")


    cmc, mAP = metrics.evaluate_locations(distmat, q_pids, g_pids, q_camids, g_camids, mode='DL')
    print("Location Results (different locations)----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")

    cmc, mAP = metrics.evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='SS')
    print("Scale Results (same scales)----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")
    cmc, mAP = metrics.evaluate_scales(distmat, q_pids, g_pids, q_camids, g_camids, mode='DS')
    print("Scale Results (different scales)----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("------------------")


    if return_distmat:
        return distmat
    return cmc[0], mAP


if __name__ == '__main__':
    main()
