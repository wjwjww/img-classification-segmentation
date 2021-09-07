import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from .base_agent import BaseModel

logger = logging.getLogger('base')


class Classification(BaseModel):
    def __init__(self, opt):
        super(Classification, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        # define network and load pre-trained models
        self.define_G()
        if self.is_train:
            self.netG.train()
            # loss
            self.define_losses()
            # optimizers and schedulers
            self.define_optimizers()
            self.define_schedulers()
            self.log_dict = OrderedDict()
        # print network
        self.print_networks()
        self.load()

    def forward(self):
        self.var_pred = self.netG(self.var_input)
        if self.opt['train']['loss']['label_smooth']:
            self.var_pred = nn.LogSoftmax(1)(self.var_pred)

    def define_entropy_loss(self, loss_type, loss_weight):
        if loss_weight > 0:
            if loss_type == 'cross':
                self.cri_entropy = nn.CrossEntropyLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_type))
            self.l_entropy_w = loss_weight
        else:
            logger.info('Remove pixel loss.')

    def cal_entropy_loss(self):
        l_g_entropy = self.l_entropy_w * self.cri_entropy(self.var_pred, self.var_label)
        self.log_dict['l_g_entropy'] = l_g_entropy.item()
        return l_g_entropy

    def define_mixup_loss(self, loss_type, loss_weight):
        # TODO
        if loss_weight > 0:
            if loss_type == 'cross':
                self.cri_entropy = nn.CrossEntropyLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] not recognized.'.format(loss_type))
            self.l_entropy_w = loss_weight
        else:
            logger.info('Remove pixel loss.')

    def cal_mixup_loss(self):
        # TODO
        l_g_entropy = self.l_entropy_w * self.cri_entropy(self.var_pred, self.var_label)
        self.log_dict['l_g_entropy'] = l_g_entropy.item()
        return l_g_entropy


class Segmentation(BaseModel):
    def __init__(self, opt):
        super(Segmentation, self).__init__(opt)
        pass

