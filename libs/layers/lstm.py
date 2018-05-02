#! /usr/bin/python
# -*- coding: utf-8 -*-

"""LSTM layer.

Layer code:
[LSTM, hidden_size, Bidirectional?, ..., Preprocessors, Postprocessors]
"""

import torch as th
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .common import Linear
from .base import ChildLayer
from .ppp import PPPSpace

__author__ = 'fyabc'


ApplyMaskInLSTM = True


class LSTMSpaceBase:
    """Search space of LSTM.

    Contains candidate values of hyperparameters.
    """

    HiddenSizes = [32, 64, 128, 256]
    UseBidirectional = [False, True]
    NumLayers = 1

    Preprocessors = PPPSpace.Preprocessors
    Postprocessors = PPPSpace.Postprocessors


class LSTMSpaceLarge(LSTMSpaceBase):
    HiddenSizes = [64, 128, 256, 512]


Spaces = {
    'base': LSTMSpaceBase,
    'large': LSTMSpaceLarge,
}


class LSTMLayer(ChildLayer):
    """The LSTM layer.

    This layer contains:
        Handle variable length inputs with masks
            See <https://zhuanlan.zhihu.com/p/28472545> for details
        Discard output states (h & c)
    """

    def __init__(self, hparams, preprocess_code, postprocess_code, *args, **kwargs):
        super().__init__(hparams, preprocess_code, postprocess_code)
        self.in_encoder = kwargs.pop('in_encoder')

        self.lstm = nn.LSTM(*args, **kwargs)

        # [NOTE]: If in decoder, requires to initialize the init state with encoder output states.
        if not self.in_encoder:
            self.fc_init_state = Linear(self.hparams.src_embedding_size, self.lstm.hidden_size)
        else:
            self.fc_init_state = None

    def forward(self, input_, lengths=None, encoder_state=None, **kwargs):
        input_ = self.preprocess(input_)

        init_h_c = self._get_init_state(input_, encoder_state)

        if not ApplyMaskInLSTM:
            return self.lstm(input_, init_h_c)[0]

        # TODO: Need test (correctness).
        if lengths is None:
            return self.lstm(input_)[0]

        _, sort_index = th.sort(-lengths)
        _, unsort_index = th.sort(sort_index)
        input_, lengths = input_[sort_index], lengths[sort_index]
        if init_h_c is not None:
            init_h_c = tuple(v[:, sort_index] for v in init_h_c)

        packed_input = nn.utils.rnn.pack_padded_sequence(input_, list(lengths.data), batch_first=True)

        # [NOTE]: Add this to disable the user warning, may reduce the memory usage.
        self.lstm.flatten_parameters()
        packed_output, _ = self.lstm(packed_input, init_h_c)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)
        output = output[unsort_index]

        # [NOTE]: The input and output sequence length must be equal.
        # In parallel training, input is split into chunks, so maximum lengths may less than global maximum,
        # So output may be shorter than input, then we should pad it.
        if output.shape[1] < input_.shape[1]:
            output = F.pad(output, (0, 0, 0, input_.shape[1] - output.shape[1], 0, 0), value=0.0)

        output = self.postprocess(output)

        return output

    def _get_init_state(self, input_, encoder_state):
        if self.in_encoder or encoder_state is None:
            return None

        batch_size = input_.size(0)
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        num_directions = 2 if self.lstm.bidirectional else 1

        # FIXME: Requires grad (connect to encoder) or not?
        h_0 = self.fc_init_state(encoder_state).unsqueeze(0).repeat(num_layers * num_directions, 1, 1)
        # FIXME: Also initialize c_0 with encoder states?
        c_0 = Variable(input_.data.new(num_layers * num_directions, batch_size, hidden_size).zero_(),
                       requires_grad=False)

        return h_0, c_0


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

    if len(layer_code) == 3:
        # Old-style layer code (without pre/post processing)
        layer_code += [0, 0]
    else:
        assert len(layer_code) == 5, 'Layer code must have length of 3 or 5, got {}'.format(len(layer_code))

    space = Spaces[hparams.lstm_space]

    batch_size, seq_length, input_size = input_shape
    hidden_size = space.HiddenSizes[layer_code[1]]
    bidirectional = space.UseBidirectional[layer_code[2]]
    num_directions = 2 if bidirectional else 1

    layer = LSTMLayer(
        hparams=hparams,
        preprocess_code=layer_code[-2],
        postprocess_code=layer_code[-1],
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=space.NumLayers,
        bias=True,
        batch_first=True,
        dropout=hparams.dropout,
        bidirectional=bidirectional,
        in_encoder=in_encoder,
    )

    return layer, th.Size([batch_size, seq_length, hidden_size * num_directions])
