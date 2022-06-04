import os
import os.path as osp
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing
import random
from PIL import ImageFilter, ImageOps, Image
from .datasets import register

def build_transform(image_size, mean, std):
    transform = create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=mean,
        std=std
    )
    return transform

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

def build_transform_weak(image_size=80,
                        interpolation='bicubic',
                        auto_augment='rand-m9-mstd0.5-inc1'):
    aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in (0.5071, 0.4866, 0.4409)]),
        )
    if interpolation and interpolation != 'random':
        aa_params['interpolation'] = Image.BICUBIC
    return transforms.Compose([
        RandomResizedCropAndInterpolation(image_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=interpolation),
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

def totensor(mean=(0.5071, 0.4866, 0.4409), std=(0.2009, 0.1984, 0.2023)):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        ])

@register('fc100')
class FC100(Dataset):

    def __init__(self, root_path, split='train', strong_prob=0.5, **kwargs):
        split_tag = split
        self.split = split
        self.strong_prob = strong_prob

        DATASET_DIR = root_path

        # Set the path according to train, val and test
        if split == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'train')
            label_list = os.listdir(THE_PATH)
        elif split == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'test')
            label_list = os.listdir(THE_PATH)
        elif split == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Unkown setname.')

        data = []
        label = []
        image_size = 80

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.n_classes = self.num_class
        mean = (0.5071, 0.4866, 0.4409)
        std = (0.2009, 0.1984, 0.2023)
        norm_params = {"mean": mean,
                       "std": std}
        transform = build_transform(image_size, mean, std)

        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if split == 'train':
            self.weak_transform = build_transform_weak(image_size=image_size)
            self.strong_transform, self.erase = build_transform_strong_part()
            self.to_tensor = totensor()
        else:
            transform = build_transform(image_size, mean, std)
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
        # Transformation
        # if split == 'train':
        #     image_size = 80
        #     self.transform = transforms.Compose([
        #         transforms.RandomResizedCrop(image_size),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])

        # else:
        #     image_size = 80
        #     self.transform = transforms.Compose([
        #         transforms.Resize([92, 92]),
        #         transforms.CenterCrop(image_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023))])

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


if __name__ == '__main__':
    pass
