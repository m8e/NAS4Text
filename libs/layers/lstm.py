#! /usr/bin/python
# -*- coding: utf-8 -*-

"""LSTM layer."""

import torch as th
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .common import Linear
from .base import ChildLayer
from .ppp import push_prepostprocessors
from ..utils.search_space import LSTMSpaces
from ..utils.common import get_reversed_index, batched_index_select

__author__ = 'fyabc'


ApplyMaskInLSTM = True


class LSTMLayer(ChildLayer):
    """The LSTM layer.

    This layer contains:
        Handle variable length inputs with masks
            See <https://zhuanlan.zhihu.com/p/28472545> for details
        Discard output states (h & c)

    [NOTE]: This layer assume that the data is left-padding.
    """

    # TODO: Add weight normalization?

    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams)
        self.in_encoder = kwargs.pop('in_encoder')
        self.reversed = kwargs.pop('reversed', False)

        if self.reversed and not self.in_encoder:
            raise RuntimeError('R2L LSTM layer only available in encoder')

        self.lstm = nn.LSTM(*args, **kwargs)
        self._init_lstm_params()

        # [NOTE]: If in decoder, requires to initialize the init state with encoder output states.
        if not self.in_encoder:
            self.fc_init_state = Linear(self.hparams.src_embedding_size, self.lstm.hidden_size, hparams=hparams)
        else:
            self.fc_init_state = None

        # [NOTE]: Only flatten parameters in single GPU training.
        # See <https://github.com/pytorch/pytorch/issues/7092> for more details.
        # TODO: Find a better solution?
        if self.hparams.distributed_world_size > 1:
            self._flatten_parameters_if_not_parallel = self._dummy
        else:
            self._flatten_parameters_if_not_parallel = self._flatten_parameters

        self.lstm.flatten_parameters()

    @property
    def batch_first(self):
        return self.lstm.batch_first

    def _dummy(self):
        pass

    def _flatten_parameters(self):
        self.lstm.flatten_parameters()

    def _init_lstm_params(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            else:
                nn.init.xavier_normal_(param)

    def forward(self, input_, lengths=None, encoder_state=None, **kwargs):
        input_ = self._reverse_io(input_, lengths=lengths)

        input_before = input_
        input_ = self.preprocess(input_)

        init_h_c = self._get_init_state(input_, encoder_state)

        if not ApplyMaskInLSTM:
            output = self.lstm(input_, init_h_c)[0]
            output = self._reverse_io(output, lengths=lengths)
            return output

        if lengths is None:
            output = self.lstm(input_)[0]
            output = self._reverse_io(output, lengths=lengths)
            return output

        _, sort_index = th.sort(-lengths)
        _, unsort_index = th.sort(sort_index)
        if self.batch_first:
            input_, lengths = input_[sort_index], lengths[sort_index]
        else:
            input_, lengths = input_[:, sort_index], lengths[sort_index]
        if init_h_c is not None:
            init_h_c = tuple(v[:, sort_index] for v in init_h_c)

        packed_input = nn.utils.rnn.pack_padded_sequence(input_, list(lengths.data), batch_first=self.batch_first)

        # [NOTE]: Add this to disable the user warning, may reduce the memory usage.
        self._flatten_parameters_if_not_parallel()

        packed_output, _ = self.lstm(packed_input, init_h_c)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=self.batch_first, padding_value=0.0)
        if self.batch_first:
            output = output[unsort_index]
        else:
            output = output[:, unsort_index]

        # [NOTE]: The input and output sequence length must be equal.
        # In parallel training, input is split into chunks, so maximum lengths may less than global maximum,
        # So output may be shorter than input, then we should pad it.
        if self.batch_first:
            in_seq_len, out_seq_len = input_.shape[1], output.shape[1]
            pad_array = (0, 0, 0, in_seq_len - out_seq_len, 0, 0)
        else:
            in_seq_len, out_seq_len = input_.shape[0], output.shape[0]
            pad_array = (0, in_seq_len - out_seq_len, 0, 0, 0, 0)
        if out_seq_len < in_seq_len:
            output = F.pad(output, pad_array, value=0.0)

        output = self.postprocess(output, input_before)

        output = self._reverse_io(output, lengths)

        return output

    def _get_init_state(self, input_, encoder_state):
        if self.in_encoder or encoder_state is None:
            return None

        batch_size = input_.size(0) if self.batch_first else input_.size(1)
        num_layers = self.lstm.num_layers
        hidden_size = self.lstm.hidden_size
        num_directions = 2 if self.lstm.bidirectional else 1

        # FIXME: Requires grad (connect to encoder) or not?
        h_0 = self.fc_init_state(encoder_state).unsqueeze(0).repeat(num_layers * num_directions, 1, 1)
        # FIXME: Also initialize c_0 with encoder states?
        c_0 = Variable(input_.data.new(num_layers * num_directions, batch_size, hidden_size).zero_(),
                       requires_grad=False)

        return h_0, c_0

    def _reverse_io(self, data, lengths):
        if not self.reversed:
            return data

        if self.batch_first:
            batch_size, max_length = data.size(0), data.size(1)
            batch_dim = 0
        else:
            max_length, batch_size = data.size(0), data.size(1)
            batch_dim = 1
        if lengths is None:
            lengths = th.full([batch_size], max_length, dtype=th.int64)

        return batched_index_select(data, get_reversed_index(lengths, max_length), dim=batch_dim)

    def extra_repr(self):
        return 'reversed={}'.format(self.reversed)


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

    space = LSTMSpaces[hparams.lstm_space]

    batch_size, seq_length, input_size = input_shape
    hidden_size = space.HiddenSizes[layer_code[1]]
    bidirectional = space.UseBidirectional[layer_code[2]]
    num_directions = 2 if bidirectional else 1

    layer = LSTMLayer(
        hparams=hparams,
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=space.NumLayers,
        bias=True,
        batch_first=not hparams.time_first,
        dropout=hparams.dropout,
        bidirectional=bidirectional,
        in_encoder=in_encoder,
    )

    output_shape = th.Size([batch_size, seq_length, hidden_size * num_directions])
    push_prepostprocessors(layer, layer_code[-2], layer_code[-1], input_shape, output_shape)

    return layer, output_shape
