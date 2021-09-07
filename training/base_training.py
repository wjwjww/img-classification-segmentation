import os
import math
import random
import logging
from collections import defaultdict

import numpy as np
import torch
from torch.backends import cudnn
from torchvision import utils

from data import create_dataloader, create_dataset
from models import create_model
from options import parse
from utils import util

cudnn.benchmark = True
# cudnn.deterministic = True


class _BaseTraining(object):
    def __init__(self, opt_path):
        self.opt_path = opt_path
        self.logger = None
        self.tb_logger = None

    def parse_args(self):
        opt = parse(self.opt_path, is_train=True)
        opt = self.add_args(opt)
        # disabled distributed training
        opt['dist'] = False
        print('Disabled distributed training.')
        self.opt = opt

    def run(self):
        self.parse_args()
        self.init_dirs()
        self.config_logger()
        self.set_tensorboard()
        self.resume_training()
        self.print_and_convert_opt()
        self.set_random_seed()
        self.create_model()
        self.create_dataloader()
        self.train()

    def add_args(self, opt):
        """if you want to add other args in 'self.opt', please override this method in subclass"""
        return opt

    def init_dirs(self):
        # mkdir, normal training
        if self.opt['path'].get('resume_state', None):
            util.mkdir_and_rename(self.opt['path']['experiments_root'], rename=False)
            # doesn't rename experiment folder when resumes, creates it if not exists
        else:
            util.mkdir_and_rename(self.opt['path']['experiments_root'],
                                  rename=True)  # rename experiment folder if exists
        util.mkdirs((path for key, path in self.opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    def config_logger(self):
        # config loggers. Before it, the log will not work
        util.setup_logger('base', self.opt['path']['log'], 'train_' + self.opt['name'], level=logging.INFO,
                          screen=True, tofile=True, propagate=False)
        self.logger = logging.getLogger('base')

    def set_tensorboard(self):
        # tensorboard logger
        if self.opt['use_tb_logger'] and 'debug' not in self.opt['name']:
            version = float(torch.__version__[0:3] + torch.__version__[4] if len(torch.__version__) > 3 else '0')
            # if version >= 1.14:  # PyTorch 1.1
            #     from torch.utils.tensorboard import SummaryWriter
            # else:
            self.logger.info(
                'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
            from tensorboardX import SummaryWriter
            self.tb_logger = SummaryWriter(log_dir=os.path.join(self.opt['path']['log'], 'tb_logger'))

    def resume_training(self):
        # loading resume state if exists
        if self.opt['path'].get('resume_state', None):
            # distributed resuming: all load into default GPU
            device_id = torch.cuda.current_device()
            resume_state = torch.load(self.opt['path']['resume_state'],
                                      map_location='cpu')
            self.check_resume(resume_state['iter'])  # check resume options
            self.logger.info('Resuming training from epoch: {}, iter: {}.'.format(
                resume_state['epoch'], resume_state['iter']))

            self.resume_state = resume_state
            self.start_epoch = resume_state['epoch']
            self.current_step = resume_state['iter']
        else:
            self.resume_state = None
            self.current_step = 0
            self.start_epoch = 0

    def check_resume(self, resume_iter):
        """Check resume states and pretrain_model paths"""
        if self.opt['path'].get('pretrain_model_G', None) is None:
            self.logger.warning('pretrain_model path will be ignored when resuming training.')
            self.opt['path']['pretrain_model_G'] = os.path.join(self.opt['path']['models'],
                                                                '{}_G.pth'.format(resume_iter))
            self.logger.info('Set [pretrain_model_G] to ' + self.opt['path']['pretrain_model_G'])
        if 'gan' in self.opt['sr_model'] and self.opt['path'].get(
                'pretrain_model_D', None) is None:
            self.opt['path']['pretrain_model_D'] = os.path.join(self.opt['path']['models'],
                                                                '{}_D.pth'.format(resume_iter))
            self.logger.info('Set [pretrain_model_D] to ' + self.opt['path']['pretrain_model_D'])

    def print_and_convert_opt(self):
        self.logger.info(util.dict2str(self.opt))  # log the configuration
        # convert to NoneDict, which returns None for missing keys
        self.opt = util.dict_to_nonedict(self.opt)

    def set_random_seed(self):
        # random seed
        seed = self.opt['train']['manual_seed']
        if seed is None:
            seed = random.randint(1, 10000)
        self.logger.info('Random seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def create_model(self):
        # create model
        self.model = create_model(self.opt)
        if self.resume_state is not None:
            self.model.resume_training(self.resume_state)       # handle optimizers and schedulers

    def create_dataloader(self):
        # create train and val dataloader
        self.create_train_data_loader()
        self.create_val_dataloader()

    def create_train_data_loader(self):
        # train dataloader
        train_data_set_opt = self.opt['datasets']['train']
        train_set = create_dataset(train_data_set_opt)
        train_size = len(train_set) // train_data_set_opt['batch_size']
        self.total_iters = int(self.opt['train']['niter'])
        self.total_epochs = int(math.ceil(self.total_iters / train_size))
        self.train_loader = create_dataloader(train_set, train_data_set_opt, self.opt)
        self.logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
        self.logger.info('Total epochs needed: {:d} for iters {:,d}'.format(self.total_epochs, self.total_iters))

    def create_val_dataloader(self):
        # val dataloader
        val_data_set_opt = self.opt['datasets']['val']
        val_set = create_dataset(val_data_set_opt)
        self.val_loader = create_dataloader(val_set, val_data_set_opt, self.opt)
        self.logger.info('Number of val images in [{:s}]: {:d}'.format(val_data_set_opt['name'], len(val_set)))

    def train(self):
        self.logger.info('Start training from epoch: {:d}, iter: {:d}'.format(self.start_epoch, self.current_step))
        for epoch in range(self.start_epoch, self.total_epochs):
            self.train_one_epoch(epoch)
        self.logger.info('Saving the final model.')
        self.model.save('latest')
        self.logger.info('End of training.')
        if self.tb_logger is not None:
            self.tb_logger.close()

    def train_one_epoch(self, epoch):
        for _, train_data in enumerate(self.train_loader):
            self.current_step += 1
            if self.current_step > self.total_iters:
                break
            self.model_step(train_data)
            self.log_training_state(epoch)
            self.validate()
            self.save_checkpoint(epoch)

    def model_step(self, train_data):
        version = float(torch.__version__[0:3])
        if version >= 1.1:
            # training
            self.model.feed_data(train_data)
            self.model.optimize_parameters(self.current_step)
            # update learning rate
            self.model.update_learning_rate(self.current_step, warmup_iter=self.opt['train']['warmup_iter'])
        else:
            # update learning rate
            self.model.update_learning_rate(self.current_step, warmup_iter=self.opt['train']['warmup_iter'])
            # training
            self.model.feed_data(train_data)
            self.model.optimize_parameters(self.current_step)

    def log_training_state(self, epoch):
        # log
        if self.current_step % self.opt['logger']['print_freq'] == 0:
            logs = self.model.get_current_log()
            message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, self.current_step)
            for v in self.model.get_current_learning_rate():
                message += '{:.3e},'.format(v)
            message += ')] '
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                # tensorboard logger
            if self.tb_logger is not None:
                self.tb_logger.add_scalars('{}|Train logs'.format(self.opt['name']), logs, self.current_step)
            self.logger.info(message)

    def validate(self):
        raise NotImplementedError

    def gather_imgs(self, visuals, val_data, imgs_dict):
        # collect all images into one dict
        LQ_paths = val_data.get('LQ_path', None)
        if LQ_paths is None:
            tem_scale = self.opt['scale'][0] if isinstance(self.opt['scale'], list) else self.opt['scale']
            LQ_paths = val_data.get('LQx{}_path'.format(tem_scale), None)
        img_name = os.path.splitext(os.path.basename(LQ_paths[0]))[0]
        imgs_dict['img_name'].append(img_name)
        for k, v in visuals.items():
            imgs_dict[k].append(v)

    def calculate_metrics(self, visuals, avg_psnrs):
        # calculate PSNR for every scale level
        scales = sorted(self.opt['scale']) if self.opt['is_multi'] else [self.opt['scale']]
        min_max = self.get_data_min_max_range()
        for s in scales:
            sr_img = util.tensor2img(visuals['SRx{}'.format(s)], min_max=min_max)  # uint8
            # sr_imgs['SRx{}'.format(s)] = sr_img
            gt_img = util.tensor2img(visuals['GTx{}'.format(s)], min_max=min_max)  # uint8

            # calculate PSNR
            sr_img, gt_img = util.crop_border([sr_img, gt_img], s)
            psnr = util.calculate_psnr(sr_img, gt_img)
            avg_psnrs['PSNR_x{}'.format(s)] += psnr
        # if is_multi, then calculate PSNR for truly SR (not progressively up-sampled)
        if self.opt['is_multi']:
            gt_img = util.tensor2img(visuals['GTx{}'.format(max(scales))], min_max=min_max)  # uint8
            for s in scales[:-1]:
                sr_img = util.tensor2img(visuals['T_SRx{}'.format(s)], min_max=min_max)  # uint8
                # calculate PSNR
                sr_img, gt_img = util.crop_border([sr_img, gt_img], s)
                psnr = util.calculate_psnr(sr_img, gt_img)
                avg_psnrs['PSNR_Tx{}'.format(s)] += psnr

    def save_images(self, all_imgs_dict):
        # save validation images
        img_names = all_imgs_dict['img_name']
        if len(img_names) > 100:
            self.save_as_each_folder(all_imgs_dict)
        else:
            try:
                self.save_as_one_img(all_imgs_dict)
            except (TypeError, RuntimeError):
                self.save_as_each_folder(all_imgs_dict)

    def get_data_min_max_range(self):
        min_max = (-1, 1) if self.opt['data_range'] == 'center' else (0, 1)
        return min_max

    def save_as_one_img(self, all_imgs_dict):
        min_max = self.get_data_min_max_range()
        for k, v in all_imgs_dict.items():
            if k != 'img_name':
                if 'GT' in k or 'LQ' in k:
                    if self.current_step == self.opt['logger']['save_sr_imgs_freq']:
                        image = torch.stack(v)
                        utils.save_image(image,
                                         os.path.join(self.opt['path']['val_images'], k + '_ref.png'),
                                         nrow=math.ceil(math.pow(image.shape[0], 0.5)),
                                         normalize=True, range=min_max)
                else:
                    if self.current_step % self.opt['logger']['save_sr_imgs_freq'] == 0:
                        img_dir = os.path.join(self.opt['path']['val_images'], k)
                        util.mkdir(img_dir)
                        save_img_path = os.path.join(img_dir, '{:d}.png'.format(self.current_step))
                        image = torch.stack(v)
                        utils.save_image(image, save_img_path, nrow=math.ceil(math.pow(image.shape[0], 0.5)),
                                         normalize=True, range=min_max)

    def save_as_each_folder(self, all_imgs_dict):
        min_max = self.get_data_min_max_range()
        if self.current_step % self.opt['logger']['save_sr_imgs_freq'] == 0:
            img_names = all_imgs_dict['img_name']
            for k, v in all_imgs_dict.items():
                if k != 'img_name' and 'GT' not in k and 'LQ' not in k:
                    for i, img_name in enumerate(img_names):
                        img = util.tensor2img(v[i], min_max=min_max)  # uint8
                        img_dir = os.path.join(self.opt['path']['val_images'], img_name + '_{}'.format(k))
                        util.mkdir(img_dir)
                        save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, self.current_step))
                        # Save SR images for reference
                        util.save_img(img, save_img_path)

    def log_validation_state(self, avg_psnrs, num):
        for k, v in avg_psnrs.items():
            avg_psnr = v / num
            avg_psnrs[k] = avg_psnr
            # log
            self.logger.info('# {}|Validation # {}: {:.4f}'.format(self.opt['name'], k, avg_psnr))
            # tensorboard logger
        if avg_psnrs:
            if self.tb_logger is not None:
                self.tb_logger.add_scalars('{}|PSNR'.format(self.opt['name']),
                                           avg_psnrs,
                                           self.current_step)

    def save_checkpoint(self, epoch):
        # save models and training states
        if self.current_step % self.opt['logger']['save_checkpoint_freq'] == 0:
            self.logger.info('Saving models and training states.')
            self.model.save(self.current_step)
            self.model.save_training_state(epoch, self.current_step)


GeneralTraining = _BaseTraining
