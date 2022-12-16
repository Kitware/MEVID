from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import torch
from .misc import get_default_video_loader
from .temporal_transforms import MultiViewTemporalTransform
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks, oids = [], [], [], []
        num_imgs = 0

        for img_paths, pid, camid, trackid, oid in data:
            num_imgs += len(img_paths)
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
            oids += [oid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        oids = set(oids)
        num_pids = len(pids)
        num_cams = len(cams)
        # num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid, oid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1], oid


class VideoDataset(Dataset):
    """Video ReID Dataset.
    Note:
        Batch data has shape N x C x T x H x W
    Args:
        dataset (list): List with items (img_paths, pid, camid)
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 dataset,
                 spatial_transform=None,
                 temporal_transform=None,
                 get_loader=get_default_video_loader):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.loader = get_loader()
        self.teacher_mode = False

    def __len__(self):
        return len(self.dataset)

    def get_num_pids(self):
        return len(np.unique([el[1] for el in self.dataset]))

    def get_num_cams(self):
        return len(np.unique([el[2] for el in self.dataset]))

    def set_teacher_mode(self, is_teacher: bool):
        self.teacher_mode = is_teacher

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        img_paths, pid, camid, tracklet_id, oid = self.dataset[index]

        if isinstance(self.temporal_transform, MultiViewTemporalTransform):
            candidates = list(filter(lambda x: x[1] == pid, self.dataset))
            img_paths = self.temporal_transform(candidates, index)
        elif self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths, index)

        clip = self.loader(img_paths)

        if not self.teacher_mode:
            clip = [self.spatial_transform(img) for img in clip]
        else:
            clip_aug = [self.spatial_transform(img) for img in clip]
            std_daug = T.Compose([
                self.spatial_transform.transforms[0],
                T.ToTensor(),
                self.spatial_transform.transforms[-1] if not isinstance(self.spatial_transform.transforms[-1], T.RandomErasing) else self.spatial_transform.transforms[-2]
            ])
            clip_std = [std_daug(img) for img in clip]
            clip = clip_aug + clip_std

        clip = torch.stack(clip, 0)

        return clip, pid, camid, tracklet_id, [i.split('/')[-1] for i in img_paths], oid
