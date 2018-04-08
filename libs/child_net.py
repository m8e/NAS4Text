#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Try to build network from code (Neural Architecture Search results)."""

import torch as th
import torch.nn as nn

from .layers.net_code import NetCodeEnum
from .layers.lstm import build_lstm
from .layers.cnn import build_cnn
from .layers.attention import build_attention

__author__ = 'fyabc'


class ChildNet(nn.Module):
    def __init__(self, net_code, hparams):
        super().__init__()

        self.net_code = net_code
        self.hparams = hparams

        # Input shape (after embedding).
        self.input_shape = th.Size([hparams.batch_size, hparams.seq_length, hparams.input_embedding_size])

        self.embedding_layer = None

        # The main network.
        self._net = []

        input_shape = self.input_shape
        for layer_code in net_code:
            layer, output_shape = self._code2layer(layer_code, input_shape)
            self._net.append(layer)
            input_shape = output_shape

        # Output shape (before softmax)
        self.output_shape = input_shape

    def forward(self, x):
        """

        Args:
            x: (batch_size, seq_len) of int32

        Returns:

        """
        for layer in self._net:
            # TODO: Fix the bug of LSTM layers (require h and c)
            x = layer(x)
        return x

    def _code2layer(self, layer_code, input_shape):
        layer_type = layer_code[0]

        if layer_type == NetCodeEnum.LSTM:
            return build_lstm(layer_code, input_shape, self.hparams)
        elif layer_type == NetCodeEnum.Convolutional:
            return build_cnn(layer_code, input_shape, self.hparams)
        elif layer_type == NetCodeEnum.Attention:
            return build_attention(layer_code, input_shape, self.hparams)
        else:
            raise ValueError('Unknown layer type {}'.format(layer_type))
