from collections import defaultdict
import numpy as np
import torch
from utils import util
from .base_training import _BaseTraining


class SegmentationTrain(_BaseTraining):
    def validate(self):
        # validation
        if self.current_step % self.opt['train']['val_freq'] == 0:
            # image restoration validation, does not support multi-GPU validation
            pbar = util.ProgressBar(len(self.val_loader))
            avg_psnrs = defaultdict(int)
            all_imgs = defaultdict(list)
            for idx, val_data in enumerate(self.val_loader):
                need_GT = False if val_data.get('GT', None) is None else True
                self.model.feed_data(val_data, need_GT=need_GT)
                self.model.test()

                visuals = self.model.get_current_visuals(need_GT=need_GT)

                if need_GT:
                    self.calculate_metrics(visuals, avg_psnrs)
                if self.current_step % self.opt['logger']['save_sr_imgs_freq'] == 0:
                    self.gather_imgs(visuals, val_data, all_imgs)
                pbar.update('Test NO.{}'.format(idx))
            if self.current_step % self.opt['logger']['save_sr_imgs_freq'] == 0:
                self.save_images(all_imgs)
            self.log_validation_state(avg_psnrs, len(self.val_loader))
