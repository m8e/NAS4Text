#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convolutional layer.

Layer code:
[CNN, OutChannels, KernelSize, Stride, ...]
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .common import Linear

__author__ = 'fyabc'


class ConvSpaceBase:
    """Search space of convolutional layers."""

    OutChannels = [8, 16, 32, 64]
    KernelSizes = [1, 3, 5, 7]
    Strides = [1, 2, 3]


class ConvSpaceLarge(ConvSpaceBase):
    OutChannels = [64, 128, 256, 512]


Spaces = {
    'base': ConvSpaceBase,
    'large': ConvSpaceLarge,
}


class EncoderConvLayer(nn.Module):
    """1D convolution layer for encoder.

    This layer contains:
        Input padding
        Residual convolutional layer for different input and output shape
        GLU gate

    Input and output have shape (B, T, C).

    Shape:
        - Input: :math:`(N, L_{in}, C_{in})`
        - Output: :math:`(N, L_{out}, C_{out})`
    """

    # [NOTE]: Since CNN may change the sequence length, how to modify the mask?
    # Current solution: assume that "stride == 1".

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

        # Residual convolutional layer for different input and output sequence length (stride > 1).
        if self.stride > 1:
            self.residual_conv = nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.residual_conv = None

    def forward(self, input_, lengths=None, **kwargs):
        x = input_.transpose(1, 2)

        # Add padding.
        padding_l = (self.conv.kernel_size[0] - 1) // 2
        padding_r = self.conv.kernel_size[0] // 2
        x = F.pad(x, (padding_l, padding_r, 0, 0, 0, 0))

        x = self.conv(x)

        # GLU.
        x = F.glu(x, dim=1)

        return x.transpose(1, 2)


class DecoderConvLayer(nn.Module):
    """1D convolution layer for decoder.

    Similar to `EncoderConvLayer`.
    Differences:
        Padding:
            Padding (kernel_size - 1) for left and right
            Remove (kernel_size - 1) last sequence from output

        Incremental state:
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
            padding=(kernel_size - 1),
        )

        # Residual convolutional layer for different input and output sequence length (stride > 1).
        if self.stride > 1:
            self.residual_conv = nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            self.residual_conv = None

    def forward(self, input_, lengths=None, **kwargs):
        x = input_.transpose(1, 2)

        x = self.conv(x)

        # GLU.
        x = F.glu(x, dim=1)

        # Remove last (kernel_size - 1) sequence
        x = x[:, :, :1 - self.kernel_size]

        return x.transpose(1, 2)


def build_cnn(layer_code, input_shape, hparams, in_encoder=True):
    """

    Args:
        layer_code:
        input_shape: torch.Size object
            Shape of input tensor, expect (batch_size, seq_len, in_channels)
        hparams:
        in_encoder: bool
            Indicates if the layer is in encoder or decoder

    Returns: layer, output_shape
    """

    # TODO: Different convolutional layer for decoder (in inference)?

    space = Spaces[hparams.conv_space]

    batch_size, seq_length, in_channels = input_shape
    out_channels = space.OutChannels[layer_code[1]]
    kernel_size = space.KernelSizes[layer_code[2]]
    stride = space.Strides[layer_code[3]]

    if in_encoder:
        layer = EncoderConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
    else:
        layer = DecoderConvLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    return layer, th.Size([batch_size, seq_length, out_channels])
