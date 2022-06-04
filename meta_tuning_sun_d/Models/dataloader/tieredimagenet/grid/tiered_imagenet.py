import os
import os.path as osp
import pickle
import random
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# from .datasets import register


# @register('tiered-imagenet')
class tieredImageNet(Dataset):

    def __init__(self, setname='train', args=None, root_path='materials/tiered-imagenet', mini=False, **kwargs):
        self.setname = setname
        split = setname
        split_tag = split
        data = np.load(os.path.join(
                root_path, '{}_images.npz'.format(split_tag)),
                allow_pickle=True)['images']
        data = data[:, :, :, ::-1]
        with open(os.path.join(
                root_path, '{}_labels.pkl'.format(split_tag)), 'rb') as f:
            label = pickle.load(f)['labels']
        
        if 'patch_list' not in vars(args).keys():
            self.patch_list = [2, 3]
            print('do not assign num_patch parameter, set as default:',self.patch_list)
        else:
            self.patch_list = args.patch_list

        if 'patch_ratio' not in vars(args).keys():
            self.patch_ratio = 2
            print('do not assign  patch_ratio parameter, set as default:',self.patch_ratio)
        else:
            self.patch_ratio = args.patch_ratio

        image_size = 80
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        if mini:
            data_ = []
            label_ = []
            np.random.seed(0)
            c = np.random.choice(max(label) + 1, 64, replace=False).tolist()
            n = len(data)
            cnt = {x: 0 for x in c}
            ind = {x: i for i, x in enumerate(c)}
            for i in range(n):
                y = int(label[i])
                if y in c and cnt[y] < 600:
                    data_.append(data[i])
                    label_.append(ind[y])
                    cnt[y] += 1
            data = data_
            label = label_

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,  
        ])
        # augment = kwargs.get('augment')
        if split == "train":
            augment = 'crop'
        else:
            augment = None
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                #transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.data)
    
    def get_grid_location(self, size, ratio, num_grid):
        '''
        :param size: size of the height/width
        :param ratio: generate grid size/ even divided grid size
        :param num_grid: number of grid
        :return: a list containing the coordinate of the grid
        '''
        raw_grid_size = int(size / num_grid)
        enlarged_grid_size = int(size / num_grid * ratio)

        center_location = raw_grid_size // 2

        location_list = []
        for i in range(num_grid):
            location_list.append((max(0, center_location - enlarged_grid_size // 2),
                                  min(size, center_location + enlarged_grid_size // 2)))
            center_location = center_location + raw_grid_size

        return location_list

    def get_pyramid(self, img, num_patch):
        if self.setname == 'val' or self.setname == 'test':
            num_grid = num_patch
            grid_ratio = self.patch_ratio

        elif self.setname == 'train':
            num_grid = num_patch
            grid_ratio = 1 + 2 * random.random()
        else:
            raise ValueError('Unkown set')

        w, h = img.size
        grid_locations_w = self.get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = self.get_grid_location(h, grid_ratio, num_grid)

        patches_list = []
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w = grid_locations_w[j]
                patch_location_h = grid_locations_h[i]
                left_up_corner_w = patch_location_w[0]
                left_up_corner_h = patch_location_h[0]
                right_down_cornet_w = patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch = img.crop((left_up_corner_w, left_up_corner_h, right_down_cornet_w, right_down_cornet_h))
                patch = self.transform(patch)
                patches_list.append(patch)

        return patches_list

    def __getitem__(self, i):
        image, label = self.data[i], self.label[i]

        # image = Image.open(path).convert('RGB')

        patch_list = []
        for num_patch in self.patch_list:
            patches = self.get_pyramid(image, num_patch)
            patch_list.extend(patches)
        patch_list = torch.stack(patch_list, dim=0)

        return patch_list, label
        # return self.transform(self.data[i]), self.label[i]


