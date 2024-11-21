import os
import numpy as np
import scipy.io as sio
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset
from utils.dataset import get_masks
from torchvision.transforms.functional import InterpolationMode


def random_crop(image_size=128, crop_scale=(0.5, 0.8)):
    return torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size, scale=crop_scale)
    )

def center_crop(image_size=128, crop_scale=0.75):
    size = int(image_size / crop_scale)
    return torch.nn.Sequential(
        transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
    )

def random_crop_val(image_size=128, crop_scale=0.75):
    size = int(image_size / crop_scale)
    return torch.nn.Sequential(
        transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomCrop(image_size),
    )

def read_hsis(label_path, root_dir):
    with open(label_path, 'r') as f:
        files = f.readlines()

    hsis = []
    for file in files:
        mat = sio.loadmat(os.path.join(root_dir, file.rstrip()))['spimg']
        mat = mat / np.max(mat)
        hsi = torch.Tensor(mat.astype(np.float32))
        hsi = torch.permute(hsi, [2, 0, 1])
        hsis.append(hsi)
    return hsis


class HSIDataset(Dataset):

    def __init__(self, label_path, root_dir, mask=get_masks(), image_size=128, repetition=10, training=True, mixup=False):
        self.hsis = read_hsis(label_path, root_dir)
        mask, _ = mask
        self.mask = torch.Tensor(mask.astype(np.float32))
        self.mask = torch.permute(self.mask, [2, 0, 1])
        self.image_size = image_size
        self.l = len(self.hsis)
        self.repetition = repetition
        self.mixup = mixup

        if training:
            self.hsi_argfcn = random_crop(image_size=image_size, crop_scale=(0.2, 0.8))
            self.mask_argfcn = random_crop(image_size=image_size, crop_scale=(0.2, 0.8))
        else:
            self.hsi_argfcn = center_crop(image_size=image_size, crop_scale=0.5)
            self.mask_argfcn = random_crop_val(image_size=image_size, crop_scale=(0.2 + 0.8) / 2.)

    def __len__(self):
        return self.l * self.repetition

    def get_one_sample(self, idx):
        idx = idx % self.l
        hsi = self.hsis[idx]

        hsi = self.hsi_argfcn(hsi)
        mask = self.mask_argfcn(self.mask)
        mask_s = torch.sum(mask, dim=0, keepdim=False)

        measure = torch.sum(hsi * mask, dim=0, keepdim=False)
        measure = measure / torch.max(measure)

        return (measure, mask, mask_s), hsi

    def __getitem__(self, item):
        (measure, mask, mask_s), hsi = self.get_one_sample(item)
        if random.randint(0, 1) == 0:
            return (measure, mask, mask_s), hsi

        if self.mixup:
            idx = random.randint(0, self.l - 1)
            (measure1, mask1, mask_s1), hsi1 = self.get_one_sample(idx)
        else:
            return (measure, mask, mask_s), hsi

        rate = random.uniform(0, 1)
        measure = measure1 * rate + measure * (1. - rate)
        mask = mask1 * rate + mask * (1. - rate)
        mask_s = mask_s1 * rate + mask_s * (1. - rate)
        hsi = hsi1 * rate + hsi * (1. - rate)

        return (measure, mask, mask_s), hsi
