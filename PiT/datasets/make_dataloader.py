import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, VideoDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, ReIDBatchSampler
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mars import Mars
from .ilids import iLIDSVID

from .misc import get_transforms, init_worker

__factory = {
    'mars': Mars,
    'ilids': iLIDSVID,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths, oids = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths, oids

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    pin_memory = True

    if cfg.DATASETS.NAMES in ['ilids', 'prid']:
        dataset_10trails = []
        for i in range(10):
            dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, split_id=i)
            dataset_10trails.append(dataset)
    else:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    if cfg.DATASETS.NAMES in ['mars', 'duke-video-reid']:
        s_tr_train, t_tr_train = get_transforms(True, cfg)
        train_set = VideoDataset(dataset.train, spatial_transform=s_tr_train,
                                 temporal_transform=t_tr_train)
    elif cfg.DATASETS.NAMES in ['ilids', 'prid']:
        s_tr_train, t_tr_train = get_transforms(True, cfg)
        train_set = [VideoDataset(i.train, spatial_transform=s_tr_train,
                                 temporal_transform=t_tr_train) for i in dataset_10trails]
    else:
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            if cfg.DATASETS.NAMES in ['mars']:
                train_loader = [DataLoader(
                    train_set, batch_sampler=ReIDBatchSampler(dataset.train, p=cfg.DATALOADER.P,
                                                              k=cfg.DATALOADER.K),
                    num_workers=num_workers, collate_fn=train_collate_fn, worker_init_fn=init_worker,
                    pin_memory=pin_memory
                )]
            elif cfg.DATASETS.NAMES in ['ilids']:
                train_loader = [DataLoader(
                    i, batch_sampler=ReIDBatchSampler(j.train, p=cfg.DATALOADER.P,
                                                              k=cfg.DATALOADER.K),
                    num_workers=num_workers, collate_fn=train_collate_fn, worker_init_fn=init_worker,
                    pin_memory=pin_memory
                ) for i,j in zip(train_set, dataset_10trails)]
            else:
                train_loader = DataLoader(
                    train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH,
                                                  cfg.DATALOADER.NUM_INSTANCE),
                    num_workers=num_workers, collate_fn=train_collate_fn
                )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    if cfg.DATASETS.NAMES in ['mars']:
        s_tr_test, t_tr_test = get_transforms(False, cfg)

        val_set = VideoDataset(dataset.query + dataset.gallery, spatial_transform=s_tr_test,
                    temporal_transform=t_tr_test)
        val_loader = [DataLoader(
            val_set, batch_size=cfg.TEST.TEST_BATCH, shuffle=False, num_workers=2,
            pin_memory=pin_memory, drop_last=False, worker_init_fn=init_worker,
            collate_fn=val_collate_fn
        )]
    elif cfg.DATASETS.NAMES in ['ilids']:
        s_tr_test, t_tr_test = get_transforms(False, cfg)

        val_set = [VideoDataset(i.query + i.gallery, spatial_transform=s_tr_test,
                               temporal_transform=t_tr_test) for i in dataset_10trails]
        val_loader = [DataLoader(
            i, batch_size=cfg.TEST.TEST_BATCH, shuffle=False, num_workers=2,
            pin_memory=pin_memory, drop_last=False, worker_init_fn=init_worker,
            collate_fn=val_collate_fn
        ) for i in val_set]
    else:
        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

        val_loader = DataLoader(
            val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )

    return train_loader, val_loader, len(dataset.query), num_classes, cam_num, view_num
