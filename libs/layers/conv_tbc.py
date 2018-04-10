#! /usr/bin/python
# -*- coding: utf-8 -*-

import math

import torch as th
import torch.nn as nn
from torch.nn.modules.utils import _single

__author__ = 'fyabc'


class _ConvTBC(nn.Module):
    """1D convolution over an input of shape (time x batch x channel)

    The implementation uses gemm to perform the convolution. This implementation
    is faster than cuDNN for small kernel sizes.
    """

    # FIXME: This is only available in recent version of PyTorch (built from source).

    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(_ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _single(kernel_size)
        self.padding = _single(padding)

        self.weight = nn.Parameter(th.Tensor(
            self.kernel_size[0], in_channels, out_channels))
        self.bias = nn.Parameter(th.Tensor(out_channels))

    def forward(self, input_):
        return input_.contiguous().conv_tbc(self.weight, self.bias, self.padding[0])

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', padding={padding}')
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    m = _ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)
