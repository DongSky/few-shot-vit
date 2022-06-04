import os
import pickle
from PIL import Image
import os.path as osp
import torch
from torch._C import device
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing
import random
from PIL import ImageFilter, ImageOps, Image
from .datasets import register
import numpy as np
# flip_and_color_jitter = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomApply(
#                 [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#                 p=0.8
#             ),
#             transforms.RandomGrayscale(p=0.2),
#         ])
# if use_prefetcher:
#     normalize = transforms.Compose([
#         transforms.ToTensor(),
#         ])
# else:
#     normalize = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])

# first global crop
# self.global_transfo1 = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
#     flip_and_color_jitter,
#     utils.GaussianBlur(1.0),
#     normalize,
# ])

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

def build_transform(image_size):
    transform = create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
    )
    return transform

def build_transform_weak(image_size=80,
                        interpolation='bicubic',
                        auto_augment='rand-m9-mstd0.5-inc1'):
    aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in (0.485, 0.456, 0.406)]),
        )
    if interpolation and interpolation != 'random':
        aa_params['interpolation'] = Image.BICUBIC
    return transforms.Compose([
        RandomResizedCropAndInterpolation((image_size, image_size), scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
                [
                    rand_augment_transform(auto_augment, aa_params),
                ],
                p=0.2),
        ])

def build_transform_strong_part(color_jitter=(0.4,0.4,0.4), re_prob=0.25, re_mode='pixel', re_count=1):
    return transforms.Compose([
        transforms.ColorJitter(*color_jitter),
        GaussianBlur(p=0.5),
        Solarization(p=0.5),
        transforms.RandomGrayscale(p=0.2),
    ]), transforms.Compose([
        RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=0, device='cpu'),
    ])

def totensor(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

@register('cub')
class CUB(Dataset):

    def __init__(self, root_path, split='train', strong_prob=0.5, **kwargs):
        setname = split
        self.split = split
        IMAGE_PATH = os.path.join(root_path, 'images')
        SPLIT_PATH = os.path.join(root_path, 'split')
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
        self.strong_prob = strong_prob
        data = []
        label = []
        lb = -1
        self.wnids = []
        image_size = 80
        if setname == 'train':
            lines.pop(5864)#this image file is broken

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(IMAGE_PATH, wnid, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.n_classes = self.num_class = np.unique(np.array(label)).shape[0]

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            #transforms.Resize(image_size),
            transforms.Resize([92, 92]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if split == 'train':
            self.weak_transform = build_transform_weak(image_size=image_size)
            self.strong_transform, self.erase = build_transform_strong_part()
            self.to_tensor = totensor()
        else:
            transform = build_transform(image_size)
            if augment == 'resize':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif augment == 'cropaug':
                self.transform = transform
                #self.transform = transforms.Compose([
                #    transforms.Resize(image_size),
                #    transforms.RandomCrop(image_size, padding=8),
                #    transforms.RandomHorizontalFlip(),
                #    transforms.ToTensor(),
                #    normalize,
                #])
            elif augment is None:
                self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = Image.open(path).convert('RGB')
        if self.split == 'train':
            weak_aug = self.weak_transform(image)
            weak_tensor = self.to_tensor(weak_aug)
            if random.random() > self.strong_prob:
                strong_aug = weak_aug
            else:
                strong_aug = self.strong_transform(weak_aug)
            strong_tensor = self.to_tensor(strong_aug)
            strong_tensor = self.erase(strong_tensor)
            return strong_tensor, weak_tensor, label
        else:
            return self.transform(image), label

