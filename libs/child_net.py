#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Try to build network from code (Neural Architecture Search results)."""

import torch as th
import torch.nn as nn

__author__ = 'fyabc'


class NetCodeEnum:
    # Layer types.
    LSTM = 0
    Convolutional = 1
    Attention = 2

    # Recurrent hyperparameters.

    # Convolutional hyperparameters.

    # Attention hyperparameters.


class ChildNet(nn.Module):
    def __init__(self, net_code):
        super().__init__()
        self._net = [self._code2layer(layer_code) for layer_code in net_code]

    def forward(self, x):
        for layer in self._net:
            x = layer(x)
        return x

    def _code2layer(self, layer_code):
        layer_type = layer_code[0]

        if layer_type == NetCodeEnum.LSTM:
            return self._build_lstm(layer_code)
        elif layer_type == NetCodeEnum.Convolutional:
            return self._build_cnn(layer_code)
        elif layer_type == NetCodeEnum.Attention:
            return self._build_attention(layer_code)
        else:
            raise ValueError('Unknown layer type {}'.format(layer_type))

    def _build_lstm(self, layer_code):
        raise NotImplementedError('LSTM layer not implemented')

    def _build_cnn(self, layer_code):
        raise NotImplementedError('Convolutional layer not implemented')

    def _build_attention(self, layer_code):
        raise NotImplementedError('Attention layer not implemented')
