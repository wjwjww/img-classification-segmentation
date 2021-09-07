# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 19:12
project: SIRE
@author: Wang Junwei
"""
import os
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from .networks import define_G
from .lr_scheduler import MultiStepLR_Restart, CosineAnnealingLR_Restart

logger = logging.getLogger('base')


# noinspection PyUnresolvedReferences
class BaseModel(object):
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data, need_label=True):
        self.var_input = data['data'].to(self.device)  # LQ
        if need_label:
            self.var_label = data['label'].to(self.device)  # GT

    def optimize_parameters(self, step):
        # G
        self.optimizer_G.zero_grad()
        self.forward()
        l_g_total = 0.
        for loss_type in self.opt['train']['loss']['G_loss']:
            l_g_total += getattr(self, 'cal_{}_loss'.format(loss_type))()
        l_g_total.backward()
        self.optimizer_G.step()

    def get_current_losses(self):
        pass

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def define_optimizers_schedulers(self):
        pass

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        # set up warm-up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        return str(network), sum(map(lambda x: x.numel(), network.parameters()))

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        """Save training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

    def define_G(self):
        self.netG = define_G(self.opt['network_G']).to(self.device)
        if self.opt['dist'] and len(self.opt['gpu_ids']) > 1:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        elif len(self.opt['gpu_ids']) > 1:
            self.netG = DataParallel(self.netG)
        else:
            print('use device: ', self.device)

    def define_optimizers(self):
        train_opt = self.opt['train']
        # optimizers
        wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
        optim_params = []
        for k, v in self.netG.named_parameters():  # can optimize for a part of the model
            if v.requires_grad:
                optim_params.append(v)
            else:
                if self.rank <= 0:
                    logger.warning('Params [{:s}] will not optimize.'.format(k))
        self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                            weight_decay=wd_G,
                                            betas=(train_opt['beta1_G'], train_opt['beta2_G']))
        self.optimizers.append(self.optimizer_G)

    def define_schedulers(self):
        train_opt = self.opt['train']
        # schedulers
        if train_opt['lr_scheme'] == 'MultiStepLR':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                        restarts=train_opt['restarts'],
                                        weights=train_opt['restart_weights'],
                                        gamma=train_opt['lr_gamma'],
                                        clear_state=train_opt['clear_state']))
        elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
            for optimizer in self.optimizers:
                self.schedulers.append(
                    CosineAnnealingLR_Restart(
                        optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                        restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
        else:
            raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

    def define_losses(self):
        loss_dict = self.opt['train']['loss']
        for k, v in loss_dict.items():
            if k.endswith('params'):
                loss_type = k.split('_')[0]
                loss_params = loss_dict['{}_params'.format(loss_type)]
                getattr(self, 'define_{}_loss'.format(loss_type))(*loss_params)

    def forward(self):
        self.var_pred = self.netG(self.var_input)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.forward()
        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_prediction(self, need_label=True):
        out_dict = OrderedDict()
        out_dict['pred'] = self.var_pred.detach().float().cpu()
        if need_label:
            out_dict['label'] = self.var_label.detach().float().cpu()
        return out_dict

    def print_network(self, net):
        s, n = self.get_network_description(net)
        if isinstance(net, nn.DataParallel) or isinstance(net, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(net.__class__.__name__,
                                             net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(net.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def print_networks(self):
        self.print_network(self.netG)
