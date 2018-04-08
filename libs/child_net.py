#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Try to build network from code (Neural Architecture Search results)."""

from itertools import chain

import torch as th
import torch.nn as nn

from .tasks import get_task
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
        self.task = get_task(hparams.task)

        # Input shape (after embedding).
        self.input_shape = th.Size([hparams.batch_size, hparams.seq_length, hparams.input_embedding_size])

        self.embedding_layer = None     # TODO

        # The main network.
        self._encoder = []
        self._decoder = []

        input_shape = self.input_shape
        for layer_code in net_code[0]:
            layer, output_shape = self._code2layer(layer_code, input_shape)
            self._encoder.append(layer)
            input_shape = output_shape

        # Intermediate shape (between encoder and decoder)
        self.intermediate_shape = input_shape

        for layer_code in net_code[0]:
            layer, output_shape = self._code2layer(layer_code, input_shape)
            self._decoder.append(layer)
            input_shape = output_shape

        # Output shape (before softmax)
        self.output_shape = input_shape

        self.softmax_layer = None

    def forward(self, x):
        """

        Args:
            x: (batch_size, seq_len) of int32

        Returns:

        """
        for layer in chain(self._encoder, self._decoder):
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
