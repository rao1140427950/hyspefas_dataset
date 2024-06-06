import os.path
import random
import cv2 as cv
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def get_masks(filepath=None):
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '../data/datasets/sensing_matrix.mat')
    mat = sio.loadmat(filepath)
    return mat['mask'], mat['mask_s']


def random_crop(image_size=128, crop_scale=(0.5, 0.8)):
    return torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(image_size, scale=crop_scale),
    )


def center_crop(image_size=128, crop_scale=0.75):
    size = int(image_size / crop_scale)
    return torch.nn.Sequential(
        transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
    )


def process_labels(label_list, equalize=True, pos_rate=None, indoor_only=False):
    positives = []
    negatives = []
    for label in label_list:
        if indoor_only:
            if not 'indoor' in label:
                continue

        label = label.rstrip().split()[0].split('.')[0]
        if label[-1] == '0':
            positives.append((label, 1))
        elif label[-1] == '1':
            negatives.append((label, 0))
        else:
            raise ValueError()
    num_p = len(positives)
    num_n = len(negatives)
    print('Found {:d} positives and {:d} negatives'.format(num_p, num_n))
    if equalize:
        if pos_rate is None:
            rate = num_n // num_p + 1
        else:
            rate = pos_rate
        positives = positives * rate
        random.shuffle(positives)
        positives = positives[:num_n]
        num_p = len(positives)
        num_n = len(negatives)
        print('Equalized to {:d} positives and {:d} negatives'.format(num_p, num_n))

    labels = positives + negatives
    random.shuffle(labels)
    return labels, positives, negatives


class SSIDataset(Dataset):

    def __init__(self, root_dir, label_path, mask=get_masks(), image_size=128, crop_scale=(0.5, 0.8), training=True,
                 cache_images=False, using_existing_cache=None, positive_rep_rate=None, indoor_only=False,
                 intra_class_mixup=False, inter_class_mixup=False):
        super(SSIDataset, self).__init__()
        f = open(label_path, 'r')
        self.labels = f.readlines()
        f.close()
        self.mask, self.mask_s = mask
        self.root_dir = root_dir
        self.indoor_only = indoor_only
        self.intra_class_mixup = intra_class_mixup
        self.inter_class_mixup = inter_class_mixup

        if training:
            self.labels, self.positives, self.negatives = process_labels(self.labels, True, positive_rep_rate, indoor_only=indoor_only)
            self.arg_fcn = random_crop(image_size, crop_scale)
        else:
            self.labels, self.positives, self.negatives = process_labels(self.labels, False, indoor_only=indoor_only)
            self.arg_fcn = center_crop(image_size, (crop_scale[0] + crop_scale[1]) / 2.)

        self.l = len(self.labels)
        self.pos_l = len(self.positives)
        self.neg_l = len(self.negatives)

        if using_existing_cache is not None:
            self.bbox_dict = using_existing_cache
        else:
            f = open(os.path.join(root_dir, '../labels.txt'), 'r')
            lines = f.readlines()
            f.close()
            self.bbox_dict = self.create_dict_from_file(lines, cache_images)

        self.mask = torch.Tensor(self.mask)
        self.mask = torch.permute(self.mask, [2, 0, 1])
        self.mask_s = torch.Tensor(self.mask_s).unsqueeze(0)

    def read_and_crop_image(self, path, coords):
        xmin, ymin, xmax, ymax = coords
        tiff = cv.imread(os.path.join(self.root_dir, path), cv.IMREAD_ANYDEPTH)
        h, w = np.shape(tiff)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w - 1, xmax)
        ymax = min(h - 1, ymax)
        return tiff[ymin:ymax, xmin:xmax], (xmin, ymin, xmax, ymax)

    def create_dict_from_file(self, content_list, cache=False):
        d = dict()
        for line in content_list:
            if self.indoor_only:
                if not 'indoor' in line:
                    continue
            name, xmin, ymin, xmax, ymax = line.split()[:5]
            xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
            xmin = round(xmin)
            ymin = round(ymin)
            xmax = round(xmax)
            ymax = round(ymax)
            if cache:
                tiff, (xmin, ymin, xmax, ymax) = self.read_and_crop_image(name, (xmin, ymin, xmax, ymax))
            else:
                tiff = None
            d[name] = tiff, (xmin, ymin, xmax, ymax)
        return d

    def __len__(self):
        return self.l

    def get_one_sample(self, idx, labels):
        path, label = labels[idx]
        path = path + '.tiff'

        tiff, (xmin, ymin, xmax, ymax) = self.bbox_dict[path]
        if tiff is None:
            tiff, (xmin, ymin, xmax, ymax) = self.read_and_crop_image(path, (xmin, ymin, xmax, ymax))
        tiff = torch.Tensor(tiff.astype(np.float32) / 16000.)

        data = torch.concat([tiff.unsqueeze(0), self.mask_s[:, ymin:ymax, xmin:xmax],
                             self.mask[:, ymin:ymax, xmin:xmax]], dim=0)
        data = self.arg_fcn(data)

        image = data[0, ...].squeeze()
        mask_s = data[1, ...].squeeze()
        mask = data[2:, ...].squeeze()
        return (image, mask, mask_s), label

    def __getitem__(self, item):
        (image, mask, mask_s), label = self.get_one_sample(item, self.labels)
        if random.randint(0, 1) == 0:
            return (image, mask, mask_s), torch.Tensor([label + 0.])

        if self.inter_class_mixup:
            idx = random.randint(0, self.l - 1)
            (image1, mask1, mask_s1), label1 = self.get_one_sample(idx, self.labels)
        elif self.intra_class_mixup:
            if label > 0.5:
                labels = self.positives
                l = self.pos_l
            else:
                labels = self.negatives
                l = self.neg_l
            idx = random.randint(0, l - 1)
            (image1, mask1, mask_s1), label1 = self.get_one_sample(idx, labels)
        else:
            return (image, mask, mask_s), torch.Tensor([label + 0.])

        rate = random.uniform(0, 1)
        image = image1 * rate + image * (1. - rate)
        mask = mask1 * rate + mask * (1. - rate)
        mask_s = mask_s1 * rate + mask_s * (1. - rate)
        label = label1 * rate + label * (1. - rate)

        return (image, mask, mask_s), torch.Tensor([label + 0.])


class RGBDataset(Dataset):

    def __init__(self, root_dir, label_path, crop_scale=(0.5, 0.8), image_size=128, training=True,
                 positive_rep_rate=None, cache_images=False, using_existing_cache=None, indoor_only=False,
                 intra_class_mixup=False, inter_class_mixup=False):
        super(RGBDataset, self).__init__()
        f = open(label_path, 'r')
        self.labels = f.readlines()
        f.close()
        self.image_size = image_size
        self.root_dir = root_dir
        self.cache_images = cache_images
        self.indoor_only = indoor_only
        self.intra_class_mixup = intra_class_mixup
        self.inter_class_mixup = inter_class_mixup

        if training:
            self.labels, self.positives, self.negatives = process_labels(self.labels, True, positive_rep_rate, indoor_only=indoor_only)
            self.arg_fcn = random_crop(image_size, crop_scale)
        else:
            self.labels, self.positives, self.negatives = process_labels(self.labels, False, indoor_only=indoor_only)
            self.arg_fcn = center_crop(image_size, (crop_scale[0] + crop_scale[1]) / 2.)

        self.l = len(self.labels)
        self.pos_l = len(self.positives)
        self.neg_l = len(self.negatives)

        if using_existing_cache is not None:
            self.images_dict = using_existing_cache
        else:
            f = open(os.path.join(root_dir, '../labels.txt'), 'r')
            lines = f.readlines()
            f.close()
            self.images_dict = self.read_and_cache_images(lines, cache=cache_images)

    def __len__(self):
        return self.l

    def read_image(self, path, coords):
        name = path.split('.')[0] + '.png'
        image = cv.imread(os.path.join(self.root_dir, name), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        return np.transpose(image, [2, 0, 1])

    def read_and_cache_images(self, content_list, cache=False):
        d = dict()
        for line in content_list:
            if self.indoor_only:
                if not 'indoor' in line:
                    continue
            name, xmin, ymin, xmax, ymax = line.split()[:5]
            xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
            if cache:
                d[name] = self.read_image(name, (xmin, ymin, xmax, ymax)), (xmin, ymin, xmax, ymax)
            else:
                d[name] =None, (xmin, ymin, xmax, ymax)
        return d

    def get_one_sample(self, idx, labels):
        path, label = labels[idx]
        path = path + '.tiff'
        rgb, _ = self.images_dict[path]
        if rgb is None:
            rgb = self.read_image(path, None)

        rgb = torch.Tensor(rgb.astype(np.float32) / 127.5 - 1.)
        rgb = self.arg_fcn(rgb)
        return rgb, label

    def __getitem__(self, item):
        rgb, label = self.get_one_sample(item, self.labels)
        if random.randint(0, 1) == 0:
            return rgb, torch.Tensor([label + 0.])

        if self.inter_class_mixup:
            idx = random.randint(0, self.l - 1)
            rgb1, label1 = self.get_one_sample(idx, self.labels)
        elif self.intra_class_mixup:
            if label > 0.5:
                labels = self.positives
                l = self.pos_l
            else:
                labels = self.negatives
                l = self.neg_l
            idx = random.randint(0, l - 1)
            rgb1, label1 = self.get_one_sample(idx, labels)
        else:
            return rgb, torch.Tensor([label + 0.])

        rate = random.uniform(0, 1)
        rgb = rgb1 * rate + rgb * (1. - rate)
        label = label1 * rate + label * (1. - rate)

        return rgb, torch.Tensor([label + 0.])


class HSIDataset(Dataset):

    def __init__(self, root_dir, label_path, image_size=128, crop_scale=(0.5, 0.8), training=False,
                 positive_rep_rate=None, indoor_only=False, cache_images=False, using_existing_cache=None,
                 intra_class_mixup=False, inter_class_mixup=False):
        super(HSIDataset, self).__init__()
        with open(label_path) as f:
            self.labels = f.readlines()
        self.root_dir = root_dir
        self.indoor_only = indoor_only
        self.intra_class_mixup = intra_class_mixup
        self.inter_class_mixup = inter_class_mixup

        if training:
            self.labels, self.positives, self.negatives = process_labels(self.labels, True, positive_rep_rate, indoor_only=indoor_only)
            self.arg_fcn = random_crop(image_size, crop_scale)
        else:
            self.labels, self.positives, self.negatives = process_labels(self.labels, False, indoor_only=indoor_only)
            self.arg_fcn = center_crop(image_size, (crop_scale[0] + crop_scale[1]) / 2.)

        self.l = len(self.labels)
        self.pos_l = len(self.positives)
        self.neg_l = len(self.negatives)

        if using_existing_cache is not None:
            self.images_dict = using_existing_cache
        else:
            f = open(os.path.join(root_dir, '../labels.txt'), 'r')
            lines = f.readlines()
            f.close()
            self.images_dict = self.read_and_cache_images(lines, cache=cache_images)

    def __len__(self):
        return self.l

    def read_image(self, path, coords):
        path = path.split('.')[0] + '.tiff.mat'
        data = sio.loadmat(os.path.join(self.root_dir, path))['var'].astype(np.float32)
        data = np.transpose(data, [2, 0, 1])
        data = data / np.max(data)
        data = data * 2. - 1.
        return data

    def read_and_cache_images(self, content_list, cache=False):
        d = dict()
        for line in content_list:
            if self.indoor_only:
                if not 'indoor' in line:
                    continue
            name, xmin, ymin, xmax, ymax = line.split()[:5]
            xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
            if cache:
                d[name] = self.read_image(name, (xmin, ymin, xmax, ymax)), (xmin, ymin, xmax, ymax)
            else:
                d[name] =None, (xmin, ymin, xmax, ymax)
        return d

    def get_one_sample(self, idx, labels):
        path, label = labels[idx]
        path = path + '.tiff'
        data, _ = self.images_dict[path]
        if data is None:
            data = self.read_image(path, None)

        data = torch.tensor(data)
        data = self.arg_fcn(data)
        return data, label

    def __getitem__(self, item):
        data, label = self.get_one_sample(item, self.labels)
        if random.randint(0, 1) == 0:
            return data, torch.tensor([label + 0.])

        if self.inter_class_mixup:
            idx = random.randint(0, self.l - 1)
            data1, label1 = self.get_one_sample(idx, self.labels)
        elif self.intra_class_mixup:
            if label > 0.5:
                labels = self.positives
                l = self.pos_l
            else:
                labels = self.negatives
                l = self.neg_l
            idx = random.randint(0, l - 1)
            data1, label1 = self.get_one_sample(idx, labels)
        else:
            return data, torch.tensor([label + 0.])

        rate = random.uniform(0, 1)
        data  = data1 * rate + data * (1. - rate)
        label = label1 * rate + label * (1. - rate)

        return data, torch.tensor([label + 0.])


class HSIRGBDataset(RGBDataset):

    def __init__(self, hsi_root_dir, rgb_root_dir, label_path, crop_scale=(0.5, 0.8), image_size=128, training=True,
                 positive_rep_rate=None, cache_images=False, using_existing_cache=None, indoor_only=False,
                 intra_class_mixup=False, inter_class_mixup=False):
        self.hsi_root_dir = hsi_root_dir
        super(HSIRGBDataset, self).__init__(
            root_dir=rgb_root_dir,
            label_path=label_path,
            crop_scale=crop_scale,
            image_size=image_size,
            training=training,
            positive_rep_rate=positive_rep_rate,
            cache_images=cache_images,
            using_existing_cache=using_existing_cache,
            indoor_only=indoor_only,
            intra_class_mixup=intra_class_mixup,
            inter_class_mixup=inter_class_mixup,
        )

    def read_image(self, path, coords):
        name = path.split('.')[0] + '.png'
        xmin, ymin, xmax, ymax = coords
        image = cv.imread(os.path.join(self.root_dir, name), cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (xmax - xmin, ymax - ymin), interpolation=cv.INTER_LINEAR)

        path = path.split('.')[0] + '.tiff.mat'
        data = sio.loadmat(os.path.join(self.hsi_root_dir, path))['var'].astype(np.float32)
        data = np.transpose(data, [2, 0, 1])
        data = data / np.max(data)
        data = data * 2. - 1.
        return np.transpose(image, [2, 0, 1]), data

    def get_one_sample(self, idx, labels):
        path, label = labels[idx]
        path = path + '.tiff'
        data, _ = self.images_dict[path]
        if data is None:
            rgb, hsi = self.read_image(path, None)
        else:
            rgb, hsi = data
        rgb = torch.Tensor(rgb.astype(np.float32) / 127.5 - 1.)

        data = torch.tensor(hsi)

        data = torch.concat([rgb, data], dim=0)
        data = self.arg_fcn(data)

        rgb = data[:3, ...]
        hsi = data[3:, ...]

        return (hsi, rgb), label

    def __getitem__(self, item):
        (hsi, rgb), label = self.get_one_sample(item, self.labels)
        if random.randint(0, 1) == 0:
            return (hsi, rgb), torch.Tensor([label + 0.])

        if self.inter_class_mixup:
            idx = random.randint(0, self.l - 1)
            (hsi1, rgb1), label1 = self.get_one_sample(idx, self.labels)
        elif self.intra_class_mixup:
            if label > 0.5:
                labels = self.positives
                l = self.pos_l
            else:
                labels = self.negatives
                l = self.neg_l
            idx = random.randint(0, l - 1)
            (hsi1, rgb1), label1 = self.get_one_sample(idx, labels)
        else:
            return (hsi, rgb), torch.Tensor([label + 0.])

        rate = random.uniform(0, 1)
        rgb = rgb1 * rate + rgb * (1. - rate)
        hsi = hsi1 * rate + hsi * (1. - rate)
        label = label1 * rate + label * (1. - rate)

        return (hsi, rgb), torch.Tensor([label + 0.])


class SSIRGBDataset(Dataset):

    def __init__(self, ssi_root_dir, rgb_root_dir, label_path, mask=get_masks(), image_size=128, crop_scale=(0.5, 0.8),
                 training=True, cache_images=False, using_existing_cache=None, positive_rep_rate=None, indoor_only=False,
                 intra_class_mixup=False, inter_class_mixup=False):
        super(SSIRGBDataset, self).__init__()
        f = open(label_path, 'r')
        self.labels = f.readlines()
        f.close()
        self.mask, self.mask_s = mask
        self.ssi_root_dir = ssi_root_dir
        self.rgb_root_dir = rgb_root_dir
        self.indoor_only = indoor_only
        self.intra_class_mixup = intra_class_mixup
        self.inter_class_mixup = inter_class_mixup

        if training:
            self.labels, self.positives, self.negatives = process_labels(self.labels, True, positive_rep_rate, indoor_only=indoor_only)
            self.arg_fcn = random_crop(image_size, crop_scale)
        else:
            self.labels, self.positives, self.negatives = process_labels(self.labels, False, indoor_only=indoor_only)
            self.arg_fcn = center_crop(image_size, (crop_scale[0] + crop_scale[1]) / 2.)

        self.l = len(self.labels)
        self.pos_l = len(self.positives)
        self.neg_l = len(self.negatives)

        if using_existing_cache is not None:
            self.bbox_dict = using_existing_cache
        else:
            f = open(os.path.join(ssi_root_dir, '../labels.txt'), 'r')
            lines = f.readlines()
            f.close()
            self.bbox_dict = self.create_dict_from_file(lines, cache_images)

        self.mask = torch.Tensor(self.mask)
        self.mask = torch.permute(self.mask, [2, 0, 1])
        self.mask_s = torch.Tensor(self.mask_s).unsqueeze(0)

    def read_and_crop_image(self, path, coords):
        xmin, ymin, xmax, ymax = coords
        tiff = cv.imread(os.path.join(self.ssi_root_dir, path), cv.IMREAD_ANYDEPTH)
        h, w = np.shape(tiff)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w - 1, xmax)
        ymax = min(h - 1, ymax)

        rgb_path = path.split('.')[0] + '.png'
        rgb = cv.imread(os.path.join(self.rgb_root_dir, rgb_path), cv.IMREAD_COLOR)
        rgb = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
        rgb = cv.resize(rgb, (xmax - xmin, ymax - ymin), interpolation=cv.INTER_LINEAR)
        rgb = np.transpose(rgb, [2, 0, 1])

        return tiff[ymin:ymax, xmin:xmax], rgb, (xmin, ymin, xmax, ymax)

    def create_dict_from_file(self, content_list, cache=False):
        d = dict()
        for line in content_list:
            if self.indoor_only:
                if not 'indoor' in line:
                    continue
            name, xmin, ymin, xmax, ymax = line.split()[:5]
            xmin, ymin, xmax, ymax = eval(xmin), eval(ymin), eval(xmax), eval(ymax)
            # xmin, ymin, xmax, ymax = get_crop_coords(xmin, ymin, xmax, ymax)
            xmin = round(xmin)
            ymin = round(ymin)
            xmax = round(xmax)
            ymax = round(ymax)
            if cache:
                tiff, rgb, (xmin, ymin, xmax, ymax) = self.read_and_crop_image(name, (xmin, ymin, xmax, ymax))
            else:
                tiff = None
                rgb = None
            d[name] = tiff, rgb, (xmin, ymin, xmax, ymax)
        return d

    def __len__(self):
        return self.l

    def get_one_sample(self, idx, labels):
        path, label = labels[idx]
        path = path + '.tiff'

        tiff, rgb, (xmin, ymin, xmax, ymax) = self.bbox_dict[path]
        if tiff is None or rgb is None:
            tiff, rgb, (xmin, ymin, xmax, ymax) = self.read_and_crop_image(path, (xmin, ymin, xmax, ymax))

        tiff = torch.Tensor(tiff.astype(np.float32) / 16000.)
        rgb = torch.Tensor(rgb.astype(np.float32) / 127.5 - 1.)

        data = torch.concat([rgb, tiff.unsqueeze(0), self.mask_s[:, ymin:ymax, xmin:xmax],
                             self.mask[:, ymin:ymax, xmin:xmax]], dim=0)
        data = self.arg_fcn(data)

        rgb = data[:3, ...]

        image = data[3, ...].squeeze()
        mask_s = data[4, ...].squeeze()
        mask = data[5:, ...].squeeze()

        return (image, mask, mask_s, rgb), label

    def __getitem__(self, item):
        (image, mask, mask_s, rgb), label = self.get_one_sample(item, self.labels)
        if random.randint(0, 1) == 0:
            return (image, mask, mask_s, rgb), torch.Tensor([label + 0.])

        if self.inter_class_mixup:
            idx = random.randint(0, self.l - 1)
            (image1, mask1, mask_s1, rgb1), label1 = self.get_one_sample(idx, self.labels)
        elif self.intra_class_mixup:
            if label > 0.5:
                labels = self.positives
                l = self.pos_l
            else:
                labels = self.negatives
                l = self.neg_l
            idx = random.randint(0, l - 1)
            (image1, mask1, mask_s1, rgb1), label1 = self.get_one_sample(idx, labels)
        else:
            return (image, mask, mask_s, rgb), torch.Tensor([label + 0.])

        rate = random.uniform(0, 1)
        image = image1 * rate + image * (1. - rate)
        mask = mask1 * rate + mask * (1. - rate)
        mask_s = mask_s1 * rate + mask_s * (1. - rate)
        rgb = rgb1 * rate + rgb * (1. - rate)
        label = label1 * rate + label * (1. - rate)

        return (image, mask, mask_s, rgb), torch.Tensor([label + 0.])

