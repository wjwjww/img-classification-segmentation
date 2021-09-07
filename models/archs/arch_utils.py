# -*- coding: utf-8 -*-
"""
Created on 2019/10/10 19:32
project: SIRE
@author: Wang Junwei
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def make_layer(block, n_layers, **kwargs):
    layers = []
    for _ in range(n_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64, Conv2d=nn.Conv2d):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class ResBlock(nn.Module):
    """
    A basic residual learning building block. (use PReLU instead)
    """

    def __init__(self, in_nc, out_nc, stride=1, bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, out_nc, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        if in_nc != out_nc:
            self.downsample = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=stride, bias=bias)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        # out = self.relu2(out)     # TODO:ReLU before element-wise addition
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)  # TODO:ReLU after addition operation
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck building block. (use PReLU instead)
    """

    def __init__(self, in_nc, neck_nc, out_nc, stride=1, bias=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_nc, neck_nc, kernel_size=1, stride=stride, bias=bias)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(neck_nc, neck_nc, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(neck_nc, out_nc, kernel_size=1, stride=stride, bias=bias)
        self.relu3 = nn.PReLU()
        if in_nc != out_nc:
            self.downsample = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=stride, bias=bias)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class SCConv2d(nn.Module):
    """
    self-calibrated conv2d
    reference: Improving Convolutional Networks with Self-Calibrated Convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', down_factor=4):
        super(SCConv2d, self).__init__()
        if in_channels % 2 != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % 2 != 0:
            raise ValueError('out_channels must be divisible by groups')
        half_in = int(in_channels // 2)
        half_out = int(out_channels // 2)
        self.conv_k1 = nn.Conv2d(in_channels=half_in, out_channels=half_out, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                 padding_mode=padding_mode)
        self.conv_k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=down_factor, stride=down_factor),
            nn.Conv2d(in_channels=half_in, out_channels=half_in, kernel_size=3,
                      stride=1, padding=1, dilation=1, groups=groups, bias=False,
                      padding_mode=padding_mode)
        )
        self.conv_k3 = nn.Conv2d(in_channels=half_in, out_channels=half_in, kernel_size=3,
                                 stride=1, padding=1, dilation=1, groups=groups, bias=False,
                                 padding_mode=padding_mode)
        self.conv_k4 = nn.Conv2d(in_channels=half_in, out_channels=half_out, kernel_size=kernel_size,
                                 stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
                                 padding_mode=padding_mode)

    def forward(self, x):
        B, C, H, W = x.size()
        first_half = x[:, :int(C // 2)]
        second_half = x[:, int(C // 2):]

        first_half_out = self.conv_k1(first_half)
        identity = second_half
        weights = torch.sigmoid(torch.add(identity, F.interpolate(self.conv_k2(second_half), size=(H, W),
                                                                  mode='bilinear', align_corners=False)))
        second_half_out = torch.mul(self.conv_k3(second_half), weights)  # k3 * sigmoid(identity + k2)
        second_half_out = self.conv_k4(second_half_out)  # k4
        return torch.cat([first_half_out, second_half_out], dim=1)


if __name__ == '__main__':
    test_in = torch.randn(1, 16, 64, 64)
    scconv = SCConv2d(16, 16, 3, 1, padding=1)
    test_out = scconv(test_in)
    print('...')
