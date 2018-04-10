#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convolutional layer.

Layer code:
[CNN, OutChannels, KernelSize, Stride, ...]
"""

import math

import torch as th
import torch.nn as nn

__author__ = 'fyabc'


def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from .conv_tbc import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return nn.utils.weight_norm(m, dim=2)


class ConvSpaceBase:
    """Search space of convolutional layers."""

    OutChannels = [8, 16, 32, 64]
    KernelSizes = [1, 3, 5, 7]
    Strides = [1, 2, 3]


class ConvSpaceLarge(ConvSpaceBase):
    OutChannels = [64, 128, 256, 512]


_Spaces = {
    'base': ConvSpaceBase,
    'large': ConvSpaceLarge,
}


class ConvBTC(nn.Conv1d):
    """1D convolution layer with input shape (B, T, C)."""

    def forward(self, input_):
        result = super().forward(input_.transpose(1, 2))
        return result.transpose(1, 2)


def build_cnn(layer_code, input_shape, hparams):
    """

    Args:
        layer_code:
        input_shape: torch.Size object
            Shape of input tensor, expect (batch_size, seq_len, in_channels)
        hparams:

    Returns: layer, output_shape
    """

    # TODO: Different convolutional layer for decoder (in inference)?

    space = _Spaces[hparams.conv_space]

    batch_size, seq_length, in_channels = input_shape
    out_channels = space.OutChannels[layer_code[1]]
    kernel_size = space.KernelSizes[layer_code[2]]
    stride = space.Strides[layer_code[3]]

    layer = ConvBTC(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
    )

    return layer, th.Size([batch_size, seq_length, out_channels])
