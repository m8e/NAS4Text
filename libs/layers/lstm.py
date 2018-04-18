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

__author__ = 'fyabc'


ApplyMaskInLSTM = True


class LSTMSpaceBase:
    """Search space of LSTM.

    Contains candidate values of hyperparameters.
    """

    HiddenSizes = [32, 64, 128, 256]
    UseBidirectional = [False, True]
    NumLayers = 1


class LSTMSpaceLarge(LSTMSpaceBase):
    HiddenSizes = [64, 128, 256, 512]


Spaces = {
    'base': LSTMSpaceBase,
    'large': LSTMSpaceLarge,
}


class LSTMLayer(nn.LSTM):
    """The LSTM layer.

    This layer contains:
        Handle variable length inputs with masks
            See <https://zhuanlan.zhihu.com/p/28472545> for details
        Discard output states (h & c)
    """
    def forward(self, input_, lengths=None):
        if not ApplyMaskInLSTM:
            return super().forward(input_)[0]

        # TODO: Need test (correctness).
        if lengths is None:
            return super().forward(input_)[0]

        _, sort_index = th.sort(-lengths)
        _, unsort_index = th.sort(sort_index)
        input_, lengths = input_[sort_index], lengths[sort_index]
        packed_input = nn.utils.rnn.pack_padded_sequence(input_, list(lengths.data), batch_first=True)
        packed_output, _ = super().forward(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)
        output = output[unsort_index]
        return output


def build_lstm(layer_code, input_shape, hparams, in_encoder=True):
    """

    Args:
        layer_code:
        input_shape: torch.Size object
            Shape of input tensor, expect (batch_size, seq_len, input_size)
        hparams:
        in_encoder: bool
            Indicates if the layer is in encoder or decoder

    Returns: layer, output_shape
        output_shape: torch.Size object
            Shape of output tensor, (batch_size, seq_len, hidden_size * num_directions)
    """

    space = Spaces[hparams.lstm_space]

    batch_size, seq_length, input_size = input_shape
    hidden_size = space.HiddenSizes[layer_code[1]]
    bidirectional = space.UseBidirectional[layer_code[2]]
    num_directions = 2 if bidirectional else 1

    layer = LSTMLayer(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=space.NumLayers,
        bias=True,
        batch_first=True,
        dropout=hparams.dropout,
        bidirectional=bidirectional,
    )

    return layer, th.Size([batch_size, seq_length, hidden_size * num_directions])
