#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convolutional layer.

Layer code:
[CNN, OutChannels, KernelSize, Stride, ...]
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

__author__ = 'fyabc'


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


class ConvLayer(nn.Module):
    """1D convolution layer.

    This layer contains:
        Input padding
        Residual connections
        Residual convolutional layer for different input and output shape
        GLU gate

    Input and output have shape (B, T, C).

    Shape:
        - Input: :math:`(N, L_{in}, C_{in})`
        - Output: :math:`(N, L_{out}, C_{out})`
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,  # Multiply by 2 for GLU
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )

        if not self.same_length:
            self.residual_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.residual_conv = None

    @property
    def same_length(self):
        return self.stride == 1 and self.in_channels == self.out_channels

    def forward(self, input_):
        x = input_.transpose(1, 2)

        residual = x

        # Add padding.
        padding_l = (self.conv.kernel_size[0] - 1) // 2
        padding_r = self.conv.kernel_size[0] // 2
        x = F.pad(x, (padding_l, padding_r, 0, 0, 0, 0))

        x = self.conv(x)

        # GLU.
        x = F.glu(x, dim=1)

        # Residual connection.
        if not self.same_length:
            residual = self.residual_conv(residual)
        x = (x + residual) * math.sqrt(0.5)

        return x.transpose(1, 2)


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

    layer = ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
    )

    return layer, th.Size([batch_size, seq_length, out_channels])
