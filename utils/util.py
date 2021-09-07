# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 16:02
project: SIRE
@author: Wang Junwei
"""
import os
import sys
import time
import math
import torch.nn.functional as F
from datetime import datetime
import random
import logging
import numpy as np
# import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from scipy.special import gamma
# from models.archs import PNetLin


####################
# dict ops
####################
def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


####################
# model resume
####################
def check_resume(opt, resume_iter):
    """Check resume states and pretrain_model paths"""
    logger = logging.getLogger('base')
    if opt['path']['resume_state']:
        if opt['path'].get('pretrain_model_G', None) is None:
            logger.warning('pretrain_model path will be ignored when resuming training.')
        if opt['model'] in ['CycleGAN']:
            opt['path']['pretrain_model_G_U'] = os.path.join(opt['path']['models'],
                                                             '{}_G_U.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_G_U] to ' + opt['path']['pretrain_model_G_U'])
            opt['path']['pretrain_model_G_D'] = os.path.join(opt['path']['models'],
                                                             '{}_G_D.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_G_D] to ' + opt['path']['pretrain_model_G_D'])
        else:
            opt['path']['pretrain_model_G'] = os.path.join(opt['path']['models'],
                                                           '{}_G.pth'.format(resume_iter))
            logger.info('Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['sr_model'] and opt['path'].get(
                'pretrain_model_D', None) is None:
            if opt['model'] in ['CycleGAN']:
                opt['path']['pretrain_model_D_U'] = os.path.join(opt['path']['models'],
                                                                 '{}_D_U.pth'.format(resume_iter))
                logger.info('Set [pretrain_model_D_U] to ' + opt['path']['pretrain_model_D_U'])
                opt['path']['pretrain_model_D_D'] = os.path.join(opt['path']['models'],
                                                                 '{}_D_D.pth'.format(resume_iter))
                logger.info('Set [pretrain_model_D_D] to ' + opt['path']['pretrain_model_D_D'])
            else:
                opt['path']['pretrain_model_D'] = os.path.join(opt['path']['models'],
                                                               '{}_D.pth'.format(resume_iter))
                logger.info('Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])


####################
# miscellaneous
####################
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path, rename=True):
    if os.path.exists(path) and rename:
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False, propagate=True):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    lg.propagate = propagate  # if True, that will be the last logger which is checked for the existence of handlers
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


####################
# image convert
####################
def crop_border(img_list, crop_border):
    """Crop borders of images
    Args:
        img_list (list [Numpy]): HWC
        crop_border (int): crop border for each end of height and weight

    Returns:
        (list [Numpy]): cropped image list
    """
    if crop_border == 0:
        return img_list
    else:
        return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0] + 1e-5)  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def img2tensor(img, min_max=(0, 1)):
    """
    NOTE: img should be BGR order of numpy nd-array
    """
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.
    img = (min_max[1] - min_max[0]) * img + min_max[0]
    if img.shape[2] == 3:
        img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(np.ascontiguousarray(img)).float()
    return img


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def DUF_downsample(x, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code

    Args:
        x (Tensor, [B, T, C, H, W]): frames to be downsampled.
        scale (int): downsampling factor: 2 | 3 | 4.
    """

    assert scale in [2, 3, 4], 'Scale [{}] is not supported'.format(scale)

    def gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    B, T, C, H, W = x.size()
    x = x.view(-1, 1, H, W)
    pad_w, pad_h = 6 + scale * 2, 6 + scale * 2  # 6 is the pad of the gaussian filter
    r_h, r_w = 0, 0
    if scale == 3:
        r_h = 3 - (H % 3)
        r_w = 3 - (W % 3)
    x = F.pad(x, [pad_w, pad_w + r_w, pad_h, pad_h + r_h], 'reflect')

    gaussian_filter = torch.from_numpy(gkern(13, 0.4 * scale)).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(B, T, C, x.size(2), x.size(3))
    return x


def single_forward(model, inp):
    """PyTorch model forward (single test), it is just a simple warpper
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    with torch.no_grad():
        model_output = model(inp)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    output = output.data.float().cpu()
    return output


def flipx4_forward(model, inp):
    """Flip testing with X4 self ensemble, i.e., normal, flip H, flip W, flip H and W
    Args:
        model (PyTorch model)
        inp (Tensor): inputs defined by the model

    Returns:
        output (Tensor): outputs of the model. float, in CPU
    """
    # normal
    output_f = single_forward(model, inp)

    # flip W
    output = single_forward(model, torch.flip(inp, (-1,)))
    output_f = output_f + torch.flip(output, (-1,))
    # flip H
    output = single_forward(model, torch.flip(inp, (-2,)))
    output_f = output_f + torch.flip(output, (-2,))
    # flip both H and W
    output = single_forward(model, torch.flip(inp, (-2, -1)))
    output_f = output_f + torch.flip(output, (-2, -1))

    return output_f / 4


####################
# a python implementation of MATLAB 'imresize' function
####################
def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2,
                                                                            (1 < absx) & (absx <= 2))
    return f


def _contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype('f4')
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(math.ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))[0]
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresize_mex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype('f4')
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype('f4')
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresize_vec(in_img, weights, indices, dim):
    if dim == 0:
        weights = weights[..., np.newaxis]
        out_img = np.sum(weights * ((in_img[..., indices, :]).astype(np.float64)), axis=-2)
    elif dim == 1:
        out_img = np.sum(weights * (in_img[..., indices].astype(np.float64)), axis=-1)
    else:
        raise NotImplementedError

    return out_img


def resize_along_dim(img, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresize_mex(img, weights, indices, dim)
    else:
        out = imresize_vec(img, weights, indices, dim)
    return out.astype('f4')


def imresize(input_img, scale_factor=None, size=None, mode='bicubic', resize_mode="vec"):
    if mode == 'bicubic':
        kernel = cubic
    elif mode == 'bilinear':
        kernel = triangle
    else:
        raise NotImplementedError
    is_tensor = False
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.detach().cpu().numpy()
        is_tensor = True
    is_transpose = False
    if input_img.ndim == 3 and (input_img.shape[-1] == 1 or input_img.shape[-1] == 3):
        input_img = np.transpose(input_img, (2, 0, 1))
        is_transpose = True
    if input_img.ndim == 4 and (input_img.shape[-1] == 1 or input_img.shape[-1] == 3):
        raise NotImplementedError
    is_uint8 = False
    if input_img.dtype == np.uint8:
        is_uint8 = True

    kernel_width = 4.0
    # Fill scale and output_size
    in_h, in_w = input_img.shape[-2], input_img.shape[-1]
    input_size = (in_h, in_w)
    if scale_factor is not None:
        scale_factor = float(scale_factor)
        scale = [scale_factor, scale_factor]
        out_h = int(math.ceil((in_h * scale_factor)))
        out_w = int(math.ceil((in_w * scale_factor)))
        size = (out_h, out_w)
    elif size is not None:
        scale_h = 1.0 * size[0] / in_h
        scale_w = 1.0 * size[1] / in_w
        scale = [scale_h, scale_w]
    else:
        raise NotImplementedError('scalar_scale OR output_shape should be defined!')
    order = np.argsort(scale)
    weights = []
    indices = []
    for k in range(2):
        w, ind = _contributions(input_size[k], size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)

    flag_2d = False
    if input_img.ndim == 2:
        input_img = input_img[np.newaxis, :]
        flag_2d = True
    for cur_dim in order:
        input_img = resize_along_dim(input_img, cur_dim, weights[cur_dim], indices[cur_dim], resize_mode)
    if is_transpose:
        input_img = np.transpose(input_img, (1, 2, 0))
    if flag_2d:
        input_img = np.squeeze(input_img)
    if is_uint8:
        input_img = np.around(np.clip(input_img, 0, 255)).astype('u1')
    if is_tensor:
        input_img = torch.from_numpy(input_img)
    return input_img


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def calculate_lpips(img1, img2, use_gpu=False, norm=False, net_type='alex', lpips=True):
    device = torch.device('cuda' if use_gpu else 'cpu')
    # pretrained net + linear layer if lpips is True
    net_LPIPS = PNetLin(pnet_type=net_type, lpips=lpips)
    if lpips:
        import os
        pth_path = '../models/archs/LPIPS_weights/{}.pth'.format(net_type)
        pth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pth_path)
        vars_dict = torch.load(pth_path, map_location='cpu')
        new_vars_dict = {}
        for k, v in vars_dict.items():
            new_k = 'lins.' + k[3:]
            new_vars_dict[new_k] = v
        net_LPIPS.load_state_dict(new_vars_dict, strict=False)
    net_LPIPS.to(device)
    net_LPIPS.eval()  # No need to train

    if norm:
        img1 = img2tensor(img1, min_max=(-1, 1))
        img2 = img2tensor(img2, min_max=(-1, 1))
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0).to(device)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0).to(device)
    score = net_LPIPS(img1, img2).squeeze().cpu().item()
    return score

####################
# NIQE
####################
# copied from BasicSR: https://github.com/xinntao/BasicSR


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = mmcv.bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) paramters.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) *
                (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute features.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat


def niqe(img,
         mu_pris_param,
         cov_pris_param,
         gaussian_window,
         block_size_h=96,
         block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.

    For good performance, it is advisable by the official implemtation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.

    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 2, (
        'Input image must be a gray or Y (of YCbCr) image with shape (h, w).')
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(
            np.abs(
                convolve(np.square(img), gaussian_window, mode='nearest') -
                np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h //
                                      scale:(idx_h + 1) * block_size_h //
                                      scale, idx_w * block_size_w //
                                      scale:(idx_w + 1) * block_size_w //
                                      scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))
        # TODO: matlab bicubic downsample with anti-aliasing
        # for simplicity, now we use opencv instead, which will result in
        # a slight difference.
        if scale == 1:
            h, w = img.shape
            img = cv2.resize(
                img / 255., (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    cov_distparam = np.cov(distparam, rowvar=False)  # TODO: use nancov

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param),
        np.transpose((mu_pris_param - mu_distparam)))
    quality = np.sqrt(quality)

    return quality


def calculate_niqe(img, crop_border, input_order='HWC', convert_to='y'):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.

    Ref: Making a "Completely Blind" Image Quality Analyzer.
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip

    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.

    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether coverted to 'y' (of MATLAB YCbCr) or 'gray'.
            Default: 'y'.

    Returns:
        float: NIQE result.
    """

    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load('basicsr/metrics/niqe_pris_params.npz')
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

    return niqe_result


class ProgressBar(object):
    """
    A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


####################
# model ops
####################
def visualize_grads(net, current_step, save_path):
    total_params = sum(map(lambda x: x.numel(), net.parameters()))
    grads = []
    for k, v in net.named_parameters():  # can count for a part of the model
        if v.requires_grad and v.grad is not None:
            grads.append(v.grad.clone().detach().numpy().reshape(-1))
    grads = np.concatenate(grads)
    assert len(grads) == total_params
    fig = plt.figure()
    ax = fig.add_subplot(111)
    min_grad = np.min(grads)
    max_grad = np.max(grads)
    p20_grad = np.percentile(grads, 20)
    p80_grad = np.percentile(grads, 80)
    interval = 0.001 if max_grad == min_grad else 0.05 * (max_grad - min_grad)
    ax.set_xlim(min_grad - interval, max_grad + interval)
    plt.text(0.05, 0.75, 'num: {}'.format(total_params), transform=ax.transAxes, va='top', ha='left')
    plt.text(0.05, 0.95, 'min: {:.5f}'.format(min_grad), transform=ax.transAxes, va='top', ha='left')
    plt.text(0.05, 0.90, 'max: {:.5f}'.format(max_grad), transform=ax.transAxes, va='top', ha='left')
    plt.text(0.05, 0.85, '20%: {}'.format(p20_grad), transform=ax.transAxes, va='top', ha='left')
    plt.text(0.05, 0.80, '80%: {}'.format(p80_grad), transform=ax.transAxes, va='top', ha='left')
    ax.hist(grads.reshape(-1), bins=100, density=True)
    fig.savefig(os.path.join(save_path, 'distribution of grads_{}.png'.format(current_step)))


if __name__ == '__main__':
    # test logger
    # setup_logger('base', '.', 'debug_train_', level=logging.INFO, screen=True, tofile=True)
    # test imresize
    import cv2

    # pics = ['baby', 'bird', 'butterfly', 'head', 'woman']
    # pics = ['baby', 'baby', 'baby', 'baby', 'baby']
    # pics = ['butterfly']
    # scale = 1 / 4
    # test 4-d data
    # img_GT_ls = []
    # img_HR_ls = []
    # img_LR_ls = []
    # for pic in pics:
    #     img_GT = cv2.imread('C:/Users/wjw/Desktop/Set5/{}.png'.format(pic))
    #     img_HR = cv2.imread('C:/Users/wjw/Desktop/bic_HR/{}.png'.format(pic))
    #     img_LR = cv2.imread('C:/Users/wjw/Desktop/save_LR/{}.png'.format(pic))
    #     img_GT_ls.append(np.transpose(img_GT, (2, 0, 1)))
    #     img_HR_ls.append(np.transpose(img_HR, (2, 0, 1)))
    #     img_LR_ls.append(np.transpose(img_LR, (2, 0, 1)))
    # img_GTs = np.stack(img_GT_ls)
    # img_HRs = np.stack(img_HR_ls)
    # img_Lrs = np.stack(img_LR_ls)
    # rlt_lrs = imresize(img_GTs, scale).astype('f4')
    # rlt_hrs = imresize(img_Lrs, 1. / scale).astype('f4')
    #
    # for i, pic in enumerate(pics):
    #     img_GT = cv2.imread('C:/Users/wjw/Desktop/Set5/{}.png'.format(pic))
    #     img_HR = cv2.imread('C:/Users/wjw/Desktop/bic_HR/{}.png'.format(pic))
    #     img_LR = cv2.imread('C:/Users/wjw/Desktop/save_LR/{}.png'.format(pic))

        # rlt_lr = imresize(img_GT, scale).astype('f4')
        # print('down-sample result of {}'.format(pic), ' max=', (rlt_lr - img_LR).astype('f4').max(), ' min=',
        #       (rlt_lr - img_LR).astype('f4').min())
        # rlt_hr = imresize(img_LR, 1. / scale).astype('f4')
        # # out = np.around(np.clip(rlt_hr, 0, 255)).astype('u1')
        # print('up-sample result of {}'.format(pic), ' max=', (rlt_hr - img_HR).astype('f4').max(),
        #       ' min=', (rlt_hr - img_HR).astype('f4').min())
        # # test 4-d data
        # rlt_lr = imresize(img_GT, scale).astype('f4')
        # print('down-sample result of {}'.format(pic), ' max=', (rlt_lr - np.transpose(rlt_lrs[i], (1, 2, 0))).max(),
        #       ' min=', (rlt_lr - np.transpose(rlt_lrs[i], (1, 2, 0))).min())
        # rlt_hr = imresize(img_LR, 1. / scale).astype('f4')
        # # out = np.around(np.clip(rlt_hr, 0, 255)).astype('u1')
        # print('up-sample result of {}'.format(pic), ' max=', (rlt_hr - np.transpose(rlt_hrs[i], (1, 2, 0))).max(),
        #       ' min=', (rlt_hr - np.transpose(rlt_hrs[i], (1, 2, 0))).min())

        # test index for different scales
        # ind2 = imresize(img_GT, 1. / 2)
        # ind4 = imresize(img_GT, 1. / 4)
        # print('finish')
    # import time
    #
    # total_time = 0
    # for i in range(10):
    #     start_time = time.time()
    #     rlt = imresize(img, scale)
    #
    #     print(rlt[:7, :7, 0])
    #     use_time = time.time() - start_time
    #     total_time += use_time
    # print('average time: {}'.format(total_time / 10))
    # print('finish test')

    # data_path = r'C:\Users\admin\Desktop\SR_RESULTS\datasets\DIV2K_valid_HR\0855.png'
    # img_GT = cv2.imread(data_path, cv2.IMREAD_UNCHANGED)
    # img1 = imresize(img_GT, 1./8.)
    # for i in range(3):
    #     img_GT = imresize(img_GT, 1./2.)
    # a = img1 == img_GT
    # cv2.imwrite('8.png', img1, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    # cv2.imwrite('222.png', img_GT, [cv2.IMWRITE_PNG_COMPRESSION, 3])

    geo_path = r'D:\landsat8_sentinel2_common_regions\LC08_L1TP_127041_20190813_20190820_01_T1.TIF'
    import rasterio
    with rasterio.open(geo_path, 'r') as src:
        src_data = src.read()
        src_profile = src.profile
    # dst_data = imresize(src_data, scale_factor=1./3., mode='bicubic')
    dst_data = imresize(src_data, scale_factor=3, mode='bicubic')
    dst_profile = src_profile.copy()
    dst_profile['height'], dst_profile['width'] = dst_data.shape[-2], dst_data.shape[-1]
    dst_profile['transform'] = dst_profile['transform'] * dst_profile['transform'].scale(1./3., 1./3.)
    dst_profile['photometric'] = 'RGB'
    with rasterio.open(geo_path.replace('.TIF', '_x3.TIF'), 'w', **dst_profile) as dst:
        dst.write(dst_data)
    # from data.geo_util import tif_to_png
    # tif_to_png(geo_path)
    # tif_to_png(geo_path.replace('.TIF', '_x3.TIF'))
    # import torch
    # import torch.nn.functional as F
    # res = F.interpolate(torch.from_numpy(np.expand_dims(src_data, axis=0)), scale_factor=3, mode='bilinear', align_corners=True)
    # with rasterio.open(geo_path.replace('.TIF', '_x3_torch_true.TIF'), 'w', **dst_profile) as dst:
    #     dst.write(res.numpy()[0])


