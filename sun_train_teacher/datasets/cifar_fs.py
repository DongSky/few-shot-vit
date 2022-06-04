import os
import os.path as osp
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
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

@register('cifar-fs')
class CIFAR_FS(Dataset):

    def __init__(self, root_path, split='train', **kwargs):

        DATASET_DIR = root_path

        # Set the path according to train, val and test
        if split == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'meta-train')
            label_list = os.listdir(THE_PATH)
        elif split == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'meta-test')
            label_list = os.listdir(THE_PATH)
        elif split == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'meta-val')
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
            # transforms.RandomResizedCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,  
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'cropaug':
            # self.transform = transforms.Compose([
            #     transforms.Resize(image_size),
            #     transforms.RandomCrop(image_size, padding=8),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize,
            # ])
            self.transform = transform
            # self.transform = transform.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize,
            # ])
        elif augment is None or augment == "test":
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
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


if __name__ == '__main__':
    pass
