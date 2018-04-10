#! /usr/bin/python
# -*- coding: utf-8 -*-

"""LSTM layer.

Layer code:
[
    LSTM,
    hidden_size,
    Bidirectional?,
    ...
]
"""

import torch as th
import torch.nn as nn
from torch.autograd import Variable

__author__ = 'fyabc'


class LSTMSpaceBase:
    """Search space of LSTM.

    Contains candidate values of hyperparameters.
    """

    HiddenSizes = [32, 64, 128, 256]
    UseBidirectional = [False, True]
    NumLayers = 1


class LSTMSpaceLarge(LSTMSpaceBase):
    HiddenSizes = [64, 128, 256, 512]


class LSTMNoOutputStates(nn.LSTM):
    """The LSTM module without output states."""
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)[0]


def build_lstm(layer_code, input_shape, hparams):
    """

    Args:
        layer_code:
        input_shape: torch.Size object
            Shape of input tensor, expect (batch_size, seq_len, input_size)
        hparams:

    Returns: layer, output_shape
        output_shape: torch.Size object
            Shape of output tensor, (batch_size, seq_len, hidden_size * num_directions)
    """

    # TODO: Specify the search space in the hparams.
    space = LSTMSpaceBase

    batch_size, seq_length, input_size = input_shape
    hidden_size = space.HiddenSizes[layer_code[1]]
    bidirectional = space.UseBidirectional[layer_code[2]]
    num_directions = 2 if bidirectional else 1

    layer = LSTMNoOutputStates(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=space.NumLayers,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=bidirectional,
    )

    def _layer_with_init_states(x):
        """Apply the layer with zero-initialized initial states, and only return the outputs."""
        return layer(x, (
            Variable(th.zeros(space.NumLayers * num_directions, batch_size, hidden_size)),
            Variable(th.zeros(space.NumLayers * num_directions, batch_size, hidden_size)),
        ))[0]

    return layer, th.Size([batch_size, seq_length, hidden_size * num_directions])
