from collections import defaultdict
import numpy as np
import torch

from .base_training import _BaseTraining


class ClassificationTrain(_BaseTraining):
    def validate(self):
        # validation
        if self.current_step % self.opt['train']['val_freq'] == 0:
            # image restoration validation, does not support multi-GPU validation
            results = defaultdict(list)
            for idx, val_data in enumerate(self.val_loader):
                need_label = False if val_data.get('label', None) is None else True
                self.model.feed_data(val_data, need_label=need_label)
                self.model.test()

                result = self.model.get_current_prediction(need_label=need_label)

                results['pred'].append(result['pred'])
                if need_label:
                    results['label'].append(result['label'])

            label = torch.cat(results['label'], dim=0).numpy()
            pred = np.argmax(torch.cat(results['pred'], dim=0).numpy(), axis=1)
            accuracy = (pred == label).mean()
            self.logger.info('# {}|Validation # accuracy: {:.4f}'.format(self.opt['name'], accuracy))
            if self.tb_logger is not None:
                self.tb_logger.add_scalars('{}|Accuracy'.format(self.opt['name']), {'acc': accuracy}, self.current_step)
