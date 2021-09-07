# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 19:14
project: SIRE
@author: Wang Junwei
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


class SmoothRegularLoss(nn.Module):
    """Smooth Regularization Loss (L1)"""

    def __init__(self, device, kernel_size=3, loss='l1', mode='W'):
        super(SmoothRegularLoss, self).__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.mode = mode
        if loss == 'l1':
            self.loss = nn.L1Loss().to(self.device)
        elif loss == 'l2':
            self.loss = nn.MSELoss().to(self.device)
        else:
            raise ValueError

    def forward(self, x, y):
        in_channels = x.shape[1]
        # out_channels = y.shape[1]
        kernel_size = (self.kernel_size, self.kernel_size)
        weights = torch.rand(1, 1, *kernel_size).to(self.device)
        norm_w8 = torch.div(weights, weights.sum()).expand(in_channels, 1, *kernel_size)

        expanded_padding = (kernel_size[1] // 2, (kernel_size[1] - 1) // 2,
                            kernel_size[0] // 2, (kernel_size[0] - 1) // 2)
        moving_avrg = F.conv2d(F.pad(x, expanded_padding, mode='circular'), norm_w8, None,
                               (1, 1), (0, 0), (1, 1), in_channels)
        if self.mode == 'W':    # weighted average
            w8 = 0.5 * torch.rand(1).to(self.device)
            x = w8 * moving_avrg + (torch.ones(1).to(self.device) - w8) * x
            loss = self.loss(x, y)
        elif self.mode == 'R':    # regular
            loss = self.loss(x, y) + 0.1 * self.loss(moving_avrg, y)
        else:
            raise ValueError
        return loss


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':         # vanilla GAN
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        elif self.gan_type == 'logistic':

            def logistic_loss(input, target):
                # target is boolean
                return F.softplus(-1 * input).mean() if target else F.softplus(input).mean()

            self.loss = logistic_loss
        elif self.gan_type == 'patch-recur':
            self.loss = nn.L1Loss(reduction='mean')
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type in ['wgan-gp', 'logistic']:
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu'), const=1.):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)
        self.const = const

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        # Note: interp should be set 'requires_grad_(True)' before forward of D
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = (grad_interp + 1e-16).norm(2, dim=1)

        loss = ((grad_interp_norm - self.const)**2).mean()
        return loss


class PathLengthRegularization(nn.Module):
    """
    borrowed from StyleGAN
    """
    def __init__(self, device=torch.device('cpu'), decay=0.01):
        super(PathLengthRegularization, self).__init__()
        self.register_buffer('mean_path_length', torch.zeros(1))
        self.mean_path_length = self.mean_path_length.to(device)
        self.decay = decay

    def forward(self, fake_img, latents):
        noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
        grad = torch.autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents,
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        mean_path_length = self.mean_path_length + self.decay * (path_lengths.mean() - self.mean_path_length)
        path_penalty = (path_lengths - mean_path_length).pow(2).mean()
        self.mean_path_length = mean_path_length.detach()
        return path_penalty, path_lengths.detach().mean(), self.mean_path_length


class LabelSmoothLoss(nn.Module):

    def __init__(self, class_num, smooth=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.class_num = class_num
        self.smooth = smooth
        self.loss = nn.KLDivLoss()#
        self.true_dist = None

    def forward(self, X, target):
        assert X.size(1) == self.class_num
        true_dist = X.clone()
        true_dist.fill_(self.smooth / (self.class_num - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smooth)
        true_dist = true_dist.detach()
        self.true_dist = true_dist

        return self.loss(X, true_dist)


# a different version of gp from CycleGAN repo
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0, conditionalD=False, var_l=None):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        if conditionalD:
            interpolatesv = torch.cat([var_l, interpolatesv], dim=1)
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
