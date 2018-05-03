#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convolutional layer.

Layer code:
[CNN, OutChannels, KernelSize, Stride, ..., Preprocessors, Postprocessors]
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .base import ChildLayer
from .ppp import PPPSpace, push_prepostprocessors

__author__ = 'fyabc'


class ConvSpaceBase:
    """Search space of convolutional layers."""

    OutChannels = [8, 16, 32, 64]
    KernelSizes = [1, 3, 5, 7]
    Strides = [1, 2, 3]

    Preprocessors = PPPSpace.Preprocessors
    Postprocessors = PPPSpace.Postprocessors


class ConvSpaceLarge(ConvSpaceBase):
    OutChannels = [64, 128, 256, 512]


Spaces = {
    'base': ConvSpaceBase,
    'large': ConvSpaceLarge,
}


class ConvLayer(ChildLayer):
    """Abstract base class of 1D convolutional layer."""
    def __init__(self, hparams, in_channels, out_channels, kernel_size, stride):
        super().__init__(hparams)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, *args):
        raise NotImplementedError

    def conv_weight_norm(self, conv):
        """Apply weight normalization on the conv layer."""
        dropout = self.hparams.dropout
        std = math.sqrt((4 * (1.0 - dropout)) / (conv.kernel_size * conv.in_channels))
        conv.weight.data.normal_(mean=0, std=std)
        conv.bias.data.zero_()

        return conv


class EncoderConvLayer(ConvLayer):
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

    def __init__(self, hparams, in_channels, out_channels, kernel_size, stride):
        super().__init__(hparams, in_channels, out_channels, kernel_size, stride)

        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,  # Multiply by 2 for GLU
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.conv = self.conv_weight_norm(conv)

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
        input_ = self.preprocess(input_)

        x = input_.transpose(1, 2)

        # Add padding.
        padding_l = (self.conv.kernel_size[0] - 1) // 2
        padding_r = self.conv.kernel_size[0] // 2
        x = F.pad(x, (padding_l, padding_r, 0, 0, 0, 0))

        x = self.conv(x)

        # GLU.
        x = F.glu(x, dim=1)

        result = x.transpose(1, 2)

        return self.postprocess(result)


class DecoderConvLayer(ConvLayer):
    """1D convolution layer for decoder.

    Similar to `EncoderConvLayer`.
    Differences:
        Padding:
            Padding (kernel_size - 1) for left and right
            Remove (kernel_size - 1) last sequence from output

        Incremental state:
    """

    def __init__(self, hparams, in_channels, out_channels, kernel_size, stride):
        super().__init__(hparams, in_channels, out_channels, kernel_size, stride)

        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,  # Multiply by 2 for GLU
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1),
        )
        self.conv = self.conv_weight_norm(conv)

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
        input_ = self.preprocess(input_)

        x = input_.transpose(1, 2)

        x = self.conv(x)

        # GLU.
        x = F.glu(x, dim=1)

        # Remove last (kernel_size - 1) sequence
        x = x[:, :, :1 - self.kernel_size]

        result = x.transpose(1, 2)

        return self.postprocess(result)


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

    if len(layer_code) == 4:
        # Old-style layer code (without pre/post processing)
        layer_code += [0, 0]
    else:
        assert len(layer_code) == 6, 'Layer code must have length of 4 or 6, got {}'.format(len(layer_code))

    space = Spaces[hparams.conv_space]

    batch_size, seq_length, in_channels = input_shape
    out_channels = space.OutChannels[layer_code[1]]
    kernel_size = space.KernelSizes[layer_code[2]]
    stride = space.Strides[layer_code[3]]

    if in_encoder:
        layer = EncoderConvLayer(
            hparams=hparams,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
    else:
        layer = DecoderConvLayer(
            hparams=hparams,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    output_shape = th.Size([batch_size, seq_length, out_channels])
    push_prepostprocessors(layer, layer_code[-2], layer_code[-1], input_shape, output_shape)

    return layer, output_shape
