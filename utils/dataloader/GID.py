import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from skimage.io import imread

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Farmland=(255, 255, 0),
    Forest=(0, 0, 255),
    Meadow=(159, 129, 183),
    Water=(0, 255, 0),
)

LABEL_MAP = OrderedDict(
    Background=0,
    Building=1,
    Farmland=2,
    Forest=3,
    Meadow=4,
    Water=5,
)


def get_gid5_dataset(root, split, binary_cls=True):
    return GID(root=root, split=split, binary_cls=binary_cls)


def get_gid_dataloader(engine, config, split='train'):
    train_dataset = GID(root=config.root, split=split, binary_cls=config.binary_cls)

    train_sampler = None
    if split == 'train':
        is_shuffle = True
        batch_size = config.batch_size
    else:
        is_shuffle = False
        batch_size = 1

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=config.num_workers,
                              drop_last=True,
                              shuffle=is_shuffle,
                              pin_memory=True,
                              sampler=train_sampler)

    return train_loader, train_sampler


class GID(Dataset):
    def __init__(self, root, split='train', binary_cls=False):
        self.root = root
        assert split in ['train', 'val', 'test']
        self.file_list = list(np.loadtxt(os.path.join(root, f'{split}_filtered.txt'), dtype=str))
        self.binary_cls = binary_cls

    @staticmethod
    def normalize(img, type='zscore', mean=np.array([249.12044, 330.4458,  432.09027]),
                  std=np.array([55.279514, 46.372353, 43.879196])):
        if type == 'zscore':
            img = img - mean
            img = img / std
            return img
        elif type == 'minmax':
            img_min = img.min()
            img_max = img.max()
            img = (img - img_min) / (img_max - img_min + 1e-8)
            return img
        elif type == '01':
            img = img / 1155.0
            return img
        else:
            raise NotImplementedError

    @staticmethod
    def calculate_ndvi(red, nir):
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1, 1)
        return ndvi

    @staticmethod
    def calculate_b2o_ratio(img):
        b2o_ratio = img[..., 2] / (np.maximum(img[..., 0], img[..., 1]) + 1e-8)
        return b2o_ratio

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.root, 'output', f'{self.file_list[idx]}.npz'))['image'].astype(np.float32)
        mask = imread(os.path.join(self.root, 'output', f'{self.file_list[idx]}_m.png')).mean(axis=2).astype(np.int64)
        rgb = image[..., :3]
        nir = image[..., 3]
        ndvi = self.calculate_ndvi(rgb[..., 0], nir)
        b2o = self.calculate_b2o_ratio(rgb)
        if self.binary_cls:
            mask = np.where(mask == 5, 1, 0)

        rgb = self.normalize(rgb, type='01')
        nir = self.normalize(nir, type='minmax')
        x = np.stack([nir, ndvi, b2o], axis=-1)
        rgb = rgb.transpose(2, 0, 1)
        x = x.transpose(2, 0, 1)
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        x = torch.from_numpy(np.ascontiguousarray(x)).float()

        return dict(data=rgb, label=mask, modal_x=x, fn=str(self.file_list[idx]), n=len(self.file_list))

    def __len__(self):
        return len(self.file_list)
