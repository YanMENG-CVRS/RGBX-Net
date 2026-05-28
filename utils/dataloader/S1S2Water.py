import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import tifffile as tiff


COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Water=(0, 255, 0),
)

LABEL_MAP = OrderedDict(
    Background=0,
    Water=1,
)


def get_s1s2water_dataset(root, split):
    return S1S2Water(root=root, split=split)


def get_s1s2water_dataloader(engine, config, split='train'):
    train_dataset = S1S2Water(root=config.root, split=split)
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
                              num_workers=1,
                              drop_last=True,
                              shuffle=is_shuffle,
                              pin_memory=False,
                              sampler=train_sampler)

    return train_loader, train_sampler


class S1S2Water(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        assert split in ['train', 'val', 'test']
        self.split = split
        self.img_file_list = sorted(os.listdir(os.path.join(self.root, split, 'img')))
        self.mask_file_list = sorted(os.listdir(os.path.join(self.root, split, 'msk')))

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
        image = tiff.imread(os.path.join(self.root, self.split, 'img', self.img_file_list[idx])).astype(np.float32)
        mask = tiff.imread(os.path.join(self.root, self.split, 'msk', self.mask_file_list[idx])).astype(np.uint8)
        rgb = image[..., [2, 1, 0]]
        nir = image[..., 3]
        ndvi = self.calculate_ndvi(rgb[..., 0], nir)
        b2o = self.calculate_b2o_ratio(rgb)

        x = np.stack([nir, ndvi, b2o], axis=-1)
        rgb = rgb.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        x = x.transpose(2, 0, 1)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask.squeeze(0))).long()
        x = torch.from_numpy(np.ascontiguousarray(x)).float()

        return dict(data=rgb, label=mask, modal_x=x, fn=str(self.img_file_list[idx]), n=len(self.img_file_list))

    def __len__(self):
        return len(self.img_file_list)
