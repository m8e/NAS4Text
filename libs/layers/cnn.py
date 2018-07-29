#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convolutional layer."""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .base import ChildLayer, wrap_ppp
from .ppp import push_prepostprocessors
from ..utils.search_space import ConvolutionalSpaces

__author__ = 'fyabc'


class ConvLayer(ChildLayer):
    """Abstract base class of 1D convolutional layer."""
    def __init__(self, hparams, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__(hparams)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.residual_conv = self.build_residual_conv(out_channels, stride)

    def forward(self, *args):
        raise NotImplementedError

    def build_residual_conv(self, out_channels, stride):
        """Residual convolutional layer for different input and output sequence length (stride > 1)."""
        # TODO: Add normalization here?
        if self.stride > 1:
            return nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )
        else:
            return None

    def conv_weight_norm(self, conv):
        """Apply weight normalization on the conv layer."""
        dropout = self.hparams.dropout
        std = math.sqrt((4 * (1.0 - dropout)) / (conv.kernel_size[0] * conv.in_channels))
        conv.weight.data.normal_(mean=0, std=std)
        conv.bias.data.zero_()

        return conv

    def modify_input_before_postprocess(self, input_):
        input_ = super().modify_input_before_postprocess(input_)
        if self.stride > 1:
            input_ = self.residual_conv(input_.transpose(1, 2)).transpose(1, 2)
        return input_


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

    def __init__(self, hparams, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__(hparams, in_channels, out_channels, kernel_size, stride, groups=groups)

        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,  # Multiply by 2 for GLU
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
        )
        self.conv = self.conv_weight_norm(conv)

    @wrap_ppp
    def forward(self, input_, lengths=None, **kwargs):
        x = input_.transpose(1, 2)

        # Add padding.
        padding_l = (self.conv.kernel_size[0] - 1) // 2
        padding_r = self.conv.kernel_size[0] // 2
        x = F.pad(x, (padding_l, padding_r, 0, 0, 0, 0))

        x = self.conv(x)

        # GLU.
        x = F.glu(x, dim=1)

        result = x.transpose(1, 2)

        return result


class DecoderConvLayer(ConvLayer):
    """1D convolution layer for decoder.

    Similar to `EncoderConvLayer`.
    Differences:
        Padding:
            Padding (kernel_size - 1) for left and right
            Remove (kernel_size - 1) last sequence from output

        Incremental state:
    """

    def __init__(self, hparams, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__(hparams, in_channels, out_channels, kernel_size, stride, groups=groups)

        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * 2,  # Multiply by 2 for GLU
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1),
            groups=groups,
        )
        self.conv = self.conv_weight_norm(conv)

    @wrap_ppp
    def forward(self, input_, lengths=None, **kwargs):
        x = input_.transpose(1, 2)

        x = self.conv(x)

        # GLU.
        x = F.glu(x, dim=1)

        # Remove last (kernel_size - 1) sequence
        x = x[:, :, :1 - self.kernel_size]

        result = x.transpose(1, 2)

        return result


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

    space = ConvolutionalSpaces[hparams.conv_space]

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
            groups=1,
        )
    else:
        layer = DecoderConvLayer(
            hparams=hparams,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=1,
        )

    output_shape = th.Size([batch_size, seq_length, out_channels])
    push_prepostprocessors(layer, layer_code[-2], layer_code[-1], input_shape, output_shape)

    return layer, output_shape
