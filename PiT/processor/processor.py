import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
import torch.distributed as dist
from torch.cuda import amp
import pdb
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank, saver, num, test):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("pit.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    cls_loss_meter = AverageMeter()
    tri_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    if cfg.MODEL.DIVERSITY:
        div_loss_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    scaler = amp.GradScaler()
    isVideo = True if cfg.DATASETS.NAMES in ['mars', 'duke-video-reid', 'ilids', 'prid'] else False
    freeze_layers = ['base', 'pyramid_layer']
    freeze_epochs = cfg.SOLVER.WARMUP_EPOCHS
    freeze_or_not = cfg.MODEL.FREEZE
    # train
    for epoch in range(1, epochs + 1):
        '''if not test:
            start_time = time.time()
            cls_loss_meter.reset()
            tri_loss_meter.reset()
            if cfg.MODEL.DIVERSITY:
                div_loss_meter.reset()
            acc_meter.reset()
            scheduler.step(epoch)
            model.train()
            if freeze_or_not and epoch <= freeze_epochs:  # freeze layers for 2000 iterations
                for name, module in model.named_children():
                    if name in freeze_layers:
                        module.eval()
            for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
                optimizer.zero_grad()
                optimizer_center.zero_grad()
                img = img.to(device)
                target = vid.to(device)
                target_cam = target_cam.to(device)
                target_view = target_view.to(device)

                with amp.autocast(enabled=True):
                    score, feat, diversity = model(img, target, cam_label=target_cam, view_label=target_view )
                    ID_LOSS, TRI_LOSS = loss_fn(score, feat, target, target_cam)
                    loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    if cfg.MODEL.DIVERSITY:
                        DIV_LOSS = sum([sum(diver_loss) / len(diver_loss) for diver_loss in diversity]) / len(diversity)
                        loss += 1.0 * DIV_LOSS

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    scaler.step(optimizer_center)
                    scaler.update()

                if isinstance(score, list):
                    acc = (score[0][0].max(1)[1] == target).float().mean()
                else:
                    acc = (score.max(1)[1] == target).float().mean()

                cls_loss_meter.update(ID_LOSS.item(), img.shape[0])
                tri_loss_meter.update(TRI_LOSS.item(), img.shape[0])
                if cfg.MODEL.DIVERSITY:
                    div_loss_meter.update(DIV_LOSS.item(), img.shape[0])
                acc_meter.update(acc, 1)

                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    if cfg.MODEL.DIVERSITY:
                        logger.info(
                            "Epoch[{}] Iteration[{}/{}] cls_loss: {:.3f}, tri_loss: {:.3f}, div_loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    cls_loss_meter.avg, tri_loss_meter.avg, div_loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
                    else:
                        logger.info("Epoch[{}] Iteration[{}/{}] cls_loss: {:.3f}, tri_loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader),
                                        cls_loss_meter.avg, tri_loss_meter.avg,acc_meter.avg, scheduler._get_lr(epoch)[0]))

            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)

            saver.dump_metric_tb(cls_loss_meter.avg, epoch, f'losses', f'cls_loss')
            saver.dump_metric_tb(tri_loss_meter.avg, epoch, f'losses', f'tri_loss')
            if cfg.MODEL.DIVERSITY:
                saver.dump_metric_tb(div_loss_meter.avg, epoch, f'losses', f'div_loss')
            saver.dump_metric_tb(acc_meter.avg, epoch, f'losses', f'acc')
            saver.dump_metric_tb(optimizer.param_groups[0]['lr'], epoch, f'losses', f'lr')

            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                if isVideo:
                    num_samples = cfg.DATALOADER.P * cfg.DATALOADER.K * cfg.DATALOADER.NUM_TRAIN_IMAGES
                    logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                                .format(epoch, time_per_batch, num_samples / time_per_batch))
                else:
                    logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

            if epoch % checkpoint_period == 0:
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                   os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
                else:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, str(num+1), cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
        elif epoch != 120:
            continue'''

        evaluator.reset()
        if epoch % eval_period == 0 or epoch == 1:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _, oids) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid, oids))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _, oids) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid, oids))

                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.3%}".format(mAP))
                for r in [1, 5, 10, 20]:
                    logger.info("CMC curve, Rank-{:<3}:{:.3%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

            saver.dump_metric_tb(mAP, epoch, f'v2v', f'mAP')
            for cmc_v in [1, 5, 10, 20]:
                saver.dump_metric_tb(cmc[cmc_v-1], epoch, f'v2v', f'cmc{cmc_v}')

    return cmc, mAP


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("pit.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, reranking=cfg.TEST.RE_RANKING)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    #img_path_list = []

    for n_iter, (img, vid, camid, camids, target_view, _, oids) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid, oids))
            #img_path_list.extend(imgpath)
    
    #import pandas as pd
    #import numpy as np
    #img_path_list = np.asarray(img_path_list)
    #data = pd.DataFrame({str(i): img_path_list[:, i] for i in range(img_path_list.shape[1])})
    #data.to_csv('img_path.csv', index=True, sep=',')

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Overall Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


