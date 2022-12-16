from __future__ import absolute_import
from __future__ import division

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import *

from PIL import Image, ImageOps
import random
import math
import numbers
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class ImageData(object):
    def __init__(self, img, x=None, y=None):
        self.img = img
        self.x = x
        self.y = y


def _group_process(images, func, params):
    if isinstance(images, (tuple, list)):
        return [_group_process(img, func, params) for img in images]
    else:
        return func(images, params)


class GroupOperation(object):
    def _instance_process(self, images, params):
        raise NotImplementedError

    def _get_params(self, images):
        return None

    def __call__(self, images):
        params = self._get_params(images)
        return _group_process(images, self._instance_process, params)


class GroupToPILImage(GroupOperation, ToPILImage):
    def __init__(self, mode=None, use_flow=False):
        super(GroupToPILImage, self).__init__(mode)
        self.use_flow = use_flow

    def _instance_process(self, pic_list, params):
        if isinstance(pic_list, np.ndarray):
            if pic_list.ndim == 3:
                return self.to_pil_image(pic_list)
            elif pic_list.ndim == 4:
                return [self.to_pil_image(pic_i) for pic_i in range(pic_list.shape[0])]
            else:
                raise TypeError
        raise TypeError

    def to_pil_image(self, pic):
        if pic.shape[2] == 3:
            return ImageData(F.to_pil_image(pic, self.mode))
        elif pic.shape[2] == 1:
            return ImageData(F.to_pil_image(pic))
        elif pic.shape[2] == 5:
            if self.use_flow:
                pic_rgb = F.to_pil_image(pic[..., :3], self.mode)
                pic_x = F.to_pil_image(pic[..., 3:4])
                pic_y = F.to_pil_image(pic[..., 4:5])
                return ImageData(pic_rgb, pic_x, pic_y)
            else:
                return ImageData(F.to_pil_image(pic[..., :3], self.mode))
        else:
            raise ValueError


class GroupResize(GroupOperation, Resize):
    def _instance_process(self, img, params):
        img.img = F.resize(img.img, self.size, self.interpolation)
        if img.x is not None:
            img.x = F.resize(img.x, self.size, self.interpolation)
        if img.y is not None:
            img.y = F.resize(img.y, self.size, self.interpolation)

        return img


class GroupRandomHorizontalFlip(GroupOperation, RandomHorizontalFlip):
    def _get_params(self, images):
        if random.random() < self.p:
            return True
        else:
            return False

    def _instance_process(self, img, flip_flag):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if flip_flag:
            img.img = F.hflip(img.img)
            if img.x is not None:
                img.x = ImageOps.invert(img.x)
        return img


class GroupRandomCrop(GroupOperation):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def pad_func(self, img, params):
        if self.padding is not None:
            img.img = F.pad(img.img, self.padding, self.fill, self.padding_mode)
            if img.x is not None:
                img.x = F.pad(img.x, self.padding, self.fill, self.padding_mode)
            if img.y is not None:
                img.y = F.pad(img.y, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed and img.img.size[0] < self.size[1]:
            img.img = F.pad(img.img, (self.size[1] - img.img.size[0], 0), self.fill, self.padding_mode)
            if img.x is not None:
                img.x = F.pad(img.x, (self.size[1] - img.img.size[0], 0), self.fill, self.padding_mode)
            if img.y is not None:
                img.y = F.pad(img.y, (self.size[1] - img.img.size[0], 0), self.fill, self.padding_mode)

        if self.pad_if_needed and img.img.size[1] < self.size[0]:
            img.img = F.pad(img.img, (0, self.size[0] - img.img.size[1]), self.fill, self.padding_mode)
            if img.x is not None:
                img.x = F.pad(img.x, (0, self.size[0] - img.img.size[1]), self.fill, self.padding_mode)
            if img.y is not None:
                img.y = F.pad(img.y, (0, self.size[0] - img.img.size[1]), self.fill, self.padding_mode)

        return img

    def _get_params(self, images):
        """
        Args:
            img (PIL Image) list: Image to be cropped.
        Returns:
            PIL Image list: Cropped image.
        """
        while isinstance(images, (tuple, list)):
            images = images[0]
        img = images.img

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return self.get_params(img, self.size)

    def _instance_process(self, images, params):
        i, j, h, w = params
        img = _group_process(images, self.pad_func, None)
        img.img = F.crop(img.img, i, j, h, w)

        if img.x is not None:
            img.x = F.crop(img.x, i, j, h, w)
        if img.y is not None:
            img.y = F.crop(img.y, i, j, h, w)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class GroupToTensor(GroupOperation, ToTensor):
    def _instance_process(self, img, params):
        img.img = F.to_tensor(img.img)
        if img.x is not None:
            img.x = F.to_tensor(img.x)
        if img.y is not None:
            img.y = F.to_tensor(img.y)

        return img


class GroupNormalize(GroupOperation, Normalize):
    def _instance_process(self, image, params):
        image.img = F.normalize(image.img, self.mean[:3], self.std[:3])
        if image.x is not None:
            image.x = F.normalize(image.x, self.mean[3:4], self.std[3:4])
        if image.y is not None:
            image.y = F.normalize(image.y, self.mean[3:4], self.std[3:4])
        return image


class GroupRandom2DTranslation(GroupOperation):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def _get_params(self, images):
        if random.uniform(0, 1) > self.p:
            return None
        else:
            new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
            x_maxrange = new_width - self.width
            y_maxrange = new_height - self.height
            x1 = int(round(random.uniform(0, x_maxrange)))
            y1 = int(round(random.uniform(0, y_maxrange)))
            return new_width, new_height, x1, y1

    def _instance_process(self, img, params):
        if params is None:
            img.img = img.img.resize((self.width, self.height), self.interpolation)

            if img.x is not None:
                img.x = img.x.resize((self.width, self.height), self.interpolation)
            if img.y is not None:
                img.y = img.y.resize((self.width, self.height), self.interpolation)

        else:
            new_width, new_height, x1, y1 = params
            img.img = img.img.resize((new_width, new_height), self.interpolation)
            img.img = img.img.crop((x1, y1, x1 + self.width, y1 + self.height))

            if img.x is not None:
                img.x = img.x.resize((new_width, new_height), self.interpolation)
                img.x = img.x.crop((x1, y1, x1 + self.width, y1 + self.height))

            if img.y is not None:
                img.y = img.y.resize((new_width, new_height), self.interpolation)
                img.y = img.y.crop((x1, y1, x1 + self.width, y1 + self.height))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GroupRandomErasing(GroupOperation):
    """ Randomly selects a rectangle region in an image and erases its pixels.
            'Random Erasing Data Augmentation' by Zhong et al.
            See https://arxiv.org/pdf/1708.04896.pdf
        Args:
             probability: The probability that the Random Erasing operation will be performed.
             sl: Minimum proportion of erased area against input image.
             sh: Maximum proportion of erased area against input image.
             r1: Minimum aspect ratio of erased area.
             mean: Erasing value.
        """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.485, 0.456, 0.406]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def _instance_process(self, img, params):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.img.size()[1] * img.img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.img.size()[2] and h < img.img.size()[1]:
                x1 = random.randint(0, img.img.size()[1] - h)
                y1 = random.randint(0, img.img.size()[2] - w)
                if img.img.size()[0] == 3:
                    img.img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img.img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img.img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img.img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                if img.x is not None:
                    img.x[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                if img.y is not None:
                    img.y[0, x1:x1 + h, y1:y1 + w] = self.mean[0]

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GroupMisAlignAugment(GroupOperation):
    """
    With a probability, crop or pad part of the image to make the images misalignment.
    """
    def __init__(self, p=0.5, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def _get_params(self, images):
        if random.uniform(0, 1) > self.p:
            return None
        else:
            postion = random.choice(['up', 'bottom'])
            operation = random.choice(['crop', 'pad'])
            return postion, operation

    def _instance_process(self, img, params):
        if params is not None:
            position, operation = params
            w, h = img.img.size

            th = int(h * self.ratio)
            if position == 'up' and operation == 'crop':
                img.img = F.crop(img.img, th, 0, h - th, w)
            elif position == 'bottom' and operation == 'crop':
                img.img = F.crop(img.img, 0, 0, h - th, w)
            elif position == 'up' and operation == 'pad':
                img.img = F.pad(img.img, (0, th, 0, 0), padding_mode='edge')
            elif position == 'bottom' and operation == 'pad':
                img.img = F.pad(img.img, (0, 0, 0, th), padding_mode='edge')

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class StackTensor(object):
    def __call__(self, tensor_list):
        if isinstance(tensor_list, (tuple, list)):
            rgb_tensor = []
            flow_tensor = []
            for tensor_i in tensor_list:
                rgb_tensor.append(tensor_i.img)
                if tensor_i.x is not None and tensor_i.y is not None:
                    flow_tensor.append(torch.cat([tensor_i.x, tensor_i.y], dim=0))
            if len(tensor_list) > 1:
                rgb_tensor = torch.stack(rgb_tensor)

                if len(flow_tensor) > 1:
                    flow_tensor = torch.stack(flow_tensor)
                    return rgb_tensor, flow_tensor

                return rgb_tensor
            else:
                if len(flow_tensor) > 0:
                    return rgb_tensor[0], flow_tensor[0]
                return rgb_tensor[0]
        raise TypeError

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToSpaceBGR(object):

    def __init__(self, is_bgr=True):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255=True):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


class ElasticTransform(object):
    '''Elastic deformation of images as described in [Simard2003]_.
       [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    '''
    def __init__(self, alpha=2000, sigma=20, order=1, mode='nearest', random_state=np.random):
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.mode = mode
        self.random_state = random_state

    def __call__(self, img):
        image = np.array(img)
        shape = image.shape
        dx = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma, mode='constant', cval=0) * self.alpha
        dy = gaussian_filter((self.random_state.rand(*shape) * 2 - 1), self.sigma, mode='constant', cval=0) * self.alpha
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        distored_image = map_coordinates(image, indices, order=self.order, mode=self.mode).reshape(image.shape)
        return Image.fromarray(distored_image)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class MisAlignAugment(object):

    def __init__(self, crop_prob=0.5, ratio=0.05):
        self.crop_prob = crop_prob
        self.ratio = ratio

    def __call__(self, img):
        """
        :param img: PIL Image to be cropped
        :return: PIL Image
        """
        is_crop = random.uniform(0, 1) < self.crop_prob
        position = random.choice(['up', 'bottom'])
        operation = random.choice(['crop', 'pad'])
        ratio = self.ratio

        if is_crop:
            w, h = img.size
            th = int(h * ratio)
            if position == 'up' and operation == 'crop':
                return F.crop(img, th, 0, h - th, w)
            elif position == 'bottom' and operation == 'crop':
                return F.crop(img, 0, 0, h - th, w)
            elif position == 'up' and operation == 'pad':
                return F.pad(img, (0, th, 0, 0), padding_mode='edge')
            elif position == 'bottom' and operation == 'pad':
                return F.pad(img, (0, 0, 0, th), padding_mode='edge')
        else:
            return img


class RandomPoseAugmentation(object):
    """Random exchange the pose specific area in a video
    {0,  "Nose"}, {1,  "Neck"}, {2,  "RShoulder"}, {3,  "RElbow"}, {4,  "RWrist"},
    {5,  "LShoulder"}, {6,  "LElbow"}, {7,  "LWrist"}, {8,  "RHip"}, {9,  "RKnee"},
    {10, "RAnkle"}, {11, "LHip"}, {12, "LKnee"}, {13, "LAnkle"}, {14, "REye"},
    {15, "LEye"}, {16, "REar"}, {17, "LEar"}"""
    def __init__(self, pixels=9, threshold=0.1, num_kps=18):
        self.pixels = pixels
        self.threshold = threshold
        self.num_kps = num_kps

    def __call__(self, imgs, img_paths, img_sizes, poses):
        def get_key(path):
            if 'ilids-vid' in path:  # ilidsvid
                key = path.split('/')[-1]
            elif 'prid2011' in path:  # prid2011
                key = '-'.join(path.split('/')[-3:])
            elif 'mars' in path:  # mars
                key = path.split('/')[-1]
            else:
                raise ValueError('{} is not acceptable'.format(path))
            return key

        for attempt in range(10):
            im1 = random.randint(0, len(img_paths) - 1)
            im2 = random.randint(0, len(img_paths) - 1)
            if im1 == im2:
                continue
            try:
                pose1 = poses[get_key(img_paths[im1])]
                pose2 = poses[get_key(img_paths[im2])]
            except:
                # some poses are not accessible
                continue
            kp_index = random.randint(0, self.num_kps - 1)  # 18 keypoints are extracted
            if pose1[kp_index][2] > self.threshold and pose2[kp_index][2] > self.threshold:
                kp1 = (pose1[kp_index][:2] * imgs[im1].size(1) / img_sizes[im1][1]).astype(int)
                kp2 = (pose2[kp_index][:2] * imgs[im2].size(1) / img_sizes[im2][1]).astype(int)
            else:
                continue

            # exchange two parts
            try:
                radius = min(kp1[0], kp1[1],
                             kp2[0], kp2[1],
                             imgs[im1].size(2) - 1 - kp1[0], imgs[im1].size(1) - 1 - kp1[1],
                             imgs[im1].size(2) - 1 - kp2[0], imgs[im1].size(1) - 1 - kp2[1],
                             self.pixels)
                start1 = kp1 - radius
                start2 = kp2 - radius
                end1 = kp1 + 1 + radius
                end2 = kp2 + 1 + radius
                tmp_part = imgs[im1][:, start1[1]:end1[1], start1[0]:end1[0]].clone()
                imgs[im1][:, start1[1]:end1[1], start1[0]:end1[0]] = imgs[im2][:, start2[1]:end2[1], start2[0]:end2[0]].clone()
                imgs[im2][:, start2[1]:end2[1], start2[0]:end2[0]] = tmp_part.clone()
            except:
                continue  # value error, ignore and continue

        return imgs


if __name__ == '__main__':
    t = ElasticTransform()
    import PIL.Image
    im = PIL.Image.open('/home/wuyiming/git/reid.pytorch/data/mars/bbox_train/0001/0001C1T0001F001.jpg')
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()
    for i in range(100):
        d_im = t(im)
        plt.imshow(d_im)
        plt.show()