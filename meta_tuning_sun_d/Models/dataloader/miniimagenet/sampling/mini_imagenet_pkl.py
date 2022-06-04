import os
import pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform

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


class MiniImageNet(Dataset):

    def __init__(self, setname, args=None, root_path='materials/mini-imagenet', **kwargs):
        split = setname
        split_tag = split
        if split == 'train':
            split_tag = 'train_phase_train'
        split_file = 'miniImageNet_category_split_{}.pickle'.format(split_tag)
        with open(os.path.join(root_path, split_file), 'rb') as f:
            pack = pickle.load(f, encoding='latin1')
        data = pack['data']
        label = pack['labels']

        image_size = 80
        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]
        
        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1
        transform = build_transform(image_size)
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        normalize = transforms.Normalize(**norm_params)
        
        self.default_transform = transforms.Compose([
            transforms.Resize(image_size),
            # transforms.RandomResizedCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            #transforms.ToTensor(),
            transforms.ToTensor(),
            normalize,  
        ])
        self.num_patch = 9
        augment = 'crop' if split == 'train' else None#kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                #transforms.Resize(image_size),
                #transforms.RandomCrop(image_size, padding=8),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            #self.transform = transform
            #self.transform = transform.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.ToTensor(),
            #normalize,
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
        data, label = self.data[i], self.label[i]
        patch_list=[]
        for _ in range(self.num_patch):
            patch_list.append(self.transform(data))#Image.open(path).convert('RGB')))
        patch_list=torch.stack(patch_list,dim=0)
        return patch_list, label
        #return self.transform(self.data[i]), self.label[i]


