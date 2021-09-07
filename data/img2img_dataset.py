# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 20:36
project: SIRE
@author: Wang Junwei
"""
import random
import numpy as np
import cv2
import lmdb
import torch
from torch.utils.data import Dataset
from .util import get_image_paths, read_img, modcrop, channel_convert, imresize_np, augment
from utils.util import imresize


class Img2ImgDataset(Dataset):
    """
    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, etc) and GT image pairs.
    If only GT images are provided, generate LQ images on-the-fly.
    """

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environments for lmdb

        self.paths_GT, self.sizes_GT = get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = get_image_paths(self.data_type, opt['dataroot_LQ'])
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and (self.GT_env is None or self.LQ_env is None):
            self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']
        if self.opt['phase'] == 'train':
            assert GT_size % scale == 0, "GT size isn't divisible by scale."

        # get GT image
        GT_path = self.paths_GT[index]
        resolution = [int(s) for s in self.sizes_GT[index].split('_')
                      ] if self.data_type == 'lmdb' else None
        img_GT = read_img(self.GT_env, GT_path, resolution)      # 3-d array, img_GT.shape == _, _, 1 or 3
        if self.opt['phase'] != 'train':  # modcrop in the validation / test phase
            img_GT = modcrop(img_GT, scale)
        if self.opt['color']:  # change color space if necessary
            img_GT = channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        # get LQ image
        if self.paths_LQ:
            # get the LQ image fro the paths_LQ, which is a 3-d array with the third axis equal to 1 or 3.
            LQ_path = self.paths_LQ[index]
            resolution = [int(s) for s in self.sizes_LQ[index].split('_')
                          ] if self.data_type == 'lmdb' else None
            img_LQ = read_img(self.LQ_env, LQ_path, resolution)
            if self.opt['color']:  # change color space if necessary
                img_LQ = channel_convert(img_LQ.shape[2], self.opt['color'], [img_LQ])[0]
            GT_H, GT_W, GT_C = img_GT.shape
            LQ_H, LQ_W, LQ_C = img_LQ.shape
            assert GT_H / LQ_H == scale and GT_W / LQ_W == scale and GT_C == LQ_C, 'GT and LQ datasets mismatched.'
        else:  # down-sampling on-the-fly
            # randomly scale during training
            # if any dimension of GT image is smaller than GT_size, then this dimension will be resized to the GT_size.
            # or all dimensions of the GT image are larger than GT_size,
            # then the GT image will be resized to (rlt = (rlt // scale) * scale)
            # at last, the LQ image will resized from the resized GT image.

            # if in the validation or test phase, LQ image will be resized from the GT image directly,
            # Note that the GT image has been cropped yet.
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(img_GT, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small, resize the GT image and regain the LQ image.
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = augment([img_LQ, img_GT], self.opt['use_flip'], self.opt['use_rot'])

        if self.opt['noise']:
            img_LQ = img_LQ + np.random.normal(0, self.opt['noise'], size=img_LQ.shape)
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, ::-1]
            img_LQ = img_LQ[:, :, ::-1]
        img_GT = np.transpose(img_GT, (2, 0, 1))
        img_LQ = np.transpose(img_LQ, (2, 0, 1))
        if self.opt['resize'] is not None:
            img_GT = imresize(img_GT, size=self.opt['resize'], mode='bicubic')
        if self.opt['pre_up']:
            # Pre-upsample LQ img before fed into network
            img_LQ = imresize(img_LQ, scale_factor=scale, mode='bilinear')
        img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).float()
        if self.opt['range'] == 'center':
            img_LQ = (img_LQ - 0.5) * 2.
            img_GT = (img_GT - 0.5) * 2.
        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
