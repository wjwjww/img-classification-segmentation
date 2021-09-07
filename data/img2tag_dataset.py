# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 20:36
project: SIRE
@author: Wang Junwei
"""
import os
import sys
import random
import numpy as np
import cv2
# import lmdb
import torch
from torch.utils.data import Dataset
from .util import read_img, channel_convert, augment


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class Img2TagDataset(Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(Img2TagDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        if isinstance(opt['dataroot'], torch._six.string_classes):
            root = os.path.expanduser(opt['dataroot'])
        else:
            root = opt['dataroot']
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.samples = make_dataset(self.root, class_to_idx, opt['extensions'])
        self.targets = [s[1] for s in self.samples]

    def __getitem__(self, index):
        path, target = self.samples[index]
        img_GT = read_img(None, path)      # 3-d array, img_GT.shape == _, _, 1 or 3
        GT_size = self.opt['input_size']
        if self.opt['color']:  # change color space if necessary
            img_GT = channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        if self.opt['phase'] == 'train':
            # TODO: resize image to a fixed size before random crop?
            # if the image size is too small, resize the GT image and regain the LQ image.
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

            # randomly crop
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_GT = img_GT[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :]

            # augmentation - flip, rotate
            img_GT = augment([img_GT], self.opt['use_flip'], self.opt['use_rot'])[0]
        else:
            img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, ::-1]
        img_GT = np.transpose(img_GT, (2, 0, 1))

        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).float()
        label = torch.tensor(target).long()
        # TODO: choose the desired normalization method, the data range is [0, 1] now.
        if self.opt['range'] == 'center':
            img_GT = (img_GT - 0.5) * 2.

        return {'data': img_GT, 'label': label, 'img_path': path, 'class': self.idx_to_class[target]}

    def __len__(self):
        return len(self.targets)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(dir, class_to_idx, extensions=IMG_EXTENSIONS):
    images = []
    dir = os.path.expanduser(dir)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    def is_valid_file(x):
        return has_file_allowed_extension(x, extensions)

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images