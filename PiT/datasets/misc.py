import functools
import os
import argparse
from argparse import Namespace
import numpy as np
from PIL import Image
from torchvision import transforms as T

from .temporal_transforms import TemporalChunkCrop, TemporalRandomFrames, RandomTemporalChunkCrop, MultiViewTemporalTransform

def init_worker(worker_id):
    np.random.seed(1234 + worker_id)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def image_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def video_loader(img_paths, image_loader):
    video = []
    for image_path in img_paths:
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def get_transforms(train_mode: bool, cfg: Namespace):

    mean, var = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    input_res = {
        'mars': (256, 128),
        'ilids': (256, 128),
    }.get(cfg.DATASETS.NAMES, (224, 224))

    erase_ratio = {
        'mars': (0.3, 3.3),
        'ilids': (0.3, 3.3),
    }.get(cfg.DATASETS.NAMES, (0.7, 1.4))

    erase_scale = (0.02, 0.4)

    resize_operation = {
        'mars': T.Resize(input_res, interpolation=3),
        'ilids': T.Resize(input_res, interpolation=3),
    }.get(cfg.DATASETS.NAMES, AdaptiveResize(height=input_res[0], width=input_res[1]))

    if not train_mode:
        t_tr_test = TemporalChunkCrop(cfg.DATALOADER.NUM_TEST_IMAGES)

        s_tr_test = T.Compose([
            resize_operation,
            T.ToTensor(),
            T.Normalize(mean, var)
        ])
        return s_tr_test, t_tr_test

    tr_re = [T.RandomErasing(p=0.5, scale=erase_scale, ratio=erase_ratio)] \
        if cfg.INPUT.RE else []

    # Data augmentation
    s_tr_train = T.Compose([
            resize_operation,
            T.Pad(10),
            T.RandomCrop(input_res),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, var)
        ] + tr_re)

    if cfg.MODEL.TRAIN_STRATEGY == 'random':
        t_tr_train = TemporalRandomFrames(cfg.DATALOADER.NUM_TRAIN_IMAGES)
    elif cfg.MODEL.TRAIN_STRATEGY == 'chunk':
        t_tr_train = RandomTemporalChunkCrop(cfg.DATALOADER.NUM_TRAIN_IMAGES)
    elif cfg.MODEL.TRAIN_STRATEGY == 'temporal':
        t_tr_train = TemporalChunkCrop(cfg.DATALOADER.NUM_TRAIN_IMAGES)
    elif cfg.MODEL.TRAIN_STRATEGY == 'multiview':
        t_tr_train = MultiViewTemporalTransform(cfg.DATALOADER.NUM_TRAIN_IMAGES)
    else:
        raise ValueError

    return s_tr_train, t_tr_train


class AdaptiveResize:
    def __init__(self, width, height, interpolation=3):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    @staticmethod
    def get_padding(padding):
        if padding == 0:
            p_1, p_2 = 0, 0
        elif padding % 2 == 0:
            p_1, p_2 = padding // 2, padding // 2
        else:
            p_1, p_2 = padding // 2 + 1, padding // 2
        return p_1, p_2

    def __call__(self, img: Image.Image):
        h, w = img.height, img.width
        # resize to ensure fit in target shape
        ratio_w = self.width / w
        ratio_h = self.height / h
        ratio = min(ratio_w, ratio_h)
        new_w, new_h = map(lambda x: int(np.floor(x * ratio)), (w, h))
        img = img.resize((new_w, new_h), resample=self.interpolation)

        # compute padding
        h, w = img.height, img.width
        p_t, p_b = self.get_padding(self.height - h)
        p_l, p_r = self.get_padding(self.width - w)

        # copy into new buffer
        img = np.pad(np.asarray(img), ((p_t, p_b), (p_l, p_r), (0, 0)), mode='constant')

        return Image.fromarray(img)
