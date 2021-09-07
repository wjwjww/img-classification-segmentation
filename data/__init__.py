# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 16:09
project: SIRE
@author: Wang Junwei
"""
# create dataset and dataloader
import logging
from torch.utils.data import DataLoader
from torch.distributed import get_world_size

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True if dataset_opt['use_shuffle'] is None else dataset_opt['use_shuffle']
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, sampler=sampler, drop_last=True,
                          pin_memory=True)
    else:
        batch_size = dataset_opt.get('batch_size', 1)
        num_workers = dataset_opt.get('n_workers', 0)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=False)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'img2tag':
        from .img2tag_dataset import Img2TagDataset as D
    elif mode == 'img2img':
        from .img2img_dataset import Img2ImgDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
