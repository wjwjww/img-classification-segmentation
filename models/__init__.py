# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 15:05
project: SIRE
@author: Wang Junwei
"""
import logging
logger = logging.getLogger('base')


def create_model(opt):
    task = opt['task']
    # image restoration
    if task == 'classification':  # PSNR-oriented super resolution
        from .agents import Classification as Model
    elif task == 'segmentation':
        from .agents import Segmentation as Model
    else:
        raise NotImplementedError('Task [{:s}] not recognized.'.format(task))
    m = Model(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
