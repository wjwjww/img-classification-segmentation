# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 19:23
project: SIRE
@author: Wang Junwei
"""
import torch
from .archs import VGG16


# Generator
def define_G(opt_net):
    which_model = opt_net['which_model_G']
    # image restoration
    if which_model == 'vgg16':
        netG = VGG16()
    elif which_model == 'resnet':
        netG = VGG16()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG

