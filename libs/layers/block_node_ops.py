#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Block node ops of block child network."""

import torch as th
import torch.nn as nn

from .common import *
from .multi_head_attention import EncDecAttention
from ..utils.search_space import CellSpace

__author__ = 'fyabc'


def _get_op_arg(self, i, default=None):
    try:
        return self.op_args[i]
    except IndexError:
        return default


class BlockNodeOp(nn.Module):
    def __init__(self, op_args, input_shape, hparams=None):
        super().__init__()
        self.op_args = op_args
        self.input_shape = input_shape
        self.hparams = hparams

    def forward(self, x, lengths=None, encoder_state=None):
        raise NotImplementedError()

    @staticmethod
    def create(op_code, op_args, input_shape, in_encoder=True, hparams=None):
        if op_code == CellSpace.LSTM:
            raise NotImplementedError()
        elif op_code == CellSpace.R2L_LSTM:
            raise NotImplementedError()
        elif op_code == CellSpace.CNN:
            raise NotImplementedError()
        elif op_code == CellSpace.SelfAttention:
            raise NotImplementedError()
        elif op_code == CellSpace.GroupedCNN:
            raise NotImplementedError()
        elif op_code == CellSpace.FFN:
            op_type = FFNOp
        elif op_code == CellSpace.Identity:
            op_type = IdentityOp
        elif op_code == CellSpace.GroupedLSTM:
            raise NotImplementedError()
        elif op_code == CellSpace.EncoderAttention:
            if in_encoder:
                raise RuntimeError('Encoder attention only available in decoder')
            op_type = EncoderAttentionOp
        else:
            raise RuntimeError('Unknown op code {}'.format(op_code))
        return op_type(op_args, input_shape, hparams=hparams)


class BlockCombineNodeOp(nn.Module):
    def __init__(self, op_args, input_shape, hparams=None):
        super().__init__()
        self.op_args = op_args
        self.input_shape = input_shape
        self.hparams = hparams

    def forward(self, in1, in2, lengths=None, encoder_state=None):
        raise NotImplementedError()

    @staticmethod
    def create(op_code, op_args, input_shape, in_encoder=True, hparams=None):
        if op_code == CellSpace.Add:
            op_type = AddOp
        elif op_code == CellSpace.Concat:
            raise NotImplementedError()
        else:
            raise RuntimeError('Unknown combine op code {}'.format(op_code))
        return op_type(op_args, input_shape, hparams=hparams)


class IdentityOp(BlockNodeOp):
    def forward(self, x, lengths=None, encoder_state=None):
        return x


class FFNOp(BlockNodeOp):
    def __init__(self, op_args, input_shape, hparams=None):
        """

        :param op_args: [activation_type, have_bias]
        :param input_shape:
        """
        super().__init__(op_args, input_shape, hparams=hparams)

        # TODO: Op-args
        bias = _get_op_arg(self, 1, True)
        self.linear = Linear(input_shape[-1], input_shape[-1], bias=bias, hparams=hparams)

    def forward(self, x, lengths=None, encoder_state=None):
        return self.linear(x)


class EncoderAttentionOp(BlockNodeOp):
    # FIXME: The decoder must contains at least one encoder attention op,
    # or the encoder is not connected to the network,
    # and will raise the error of "grad is None, does not have data", etc.
    def __init__(self, op_args, input_shape, hparams=None):
        """

        :param op_args: [num_heads]
        :param input_shape:
        """
        super().__init__(op_args, input_shape, hparams=hparams)

        h = _get_op_arg(self, 0, 4)

        self.attention = EncDecAttention(
            h,
            input_shape[2],
            hparams.trg_embedding_size, hparams.src_embedding_size,
            dropout=hparams.dropout, in_encoder=False, hparams=hparams,
            linear_bias=hparams.attn_linear_bias
        )

        self.attn_scores = None

    def forward(self, x, lengths=None, encoder_state=None):
        """
        Args:
            x: (batch_size, trg_seq_len, conv_channels) of float32
            encoder_state (tuple):
                output: (batch_size, src_seq_len, src_emb_size) of float32
                output add source embedding: same shape as output
            lengths: (batch_size,) of long

        Returns:
            output: (batch_size, trg_seq_len, conv_channels) of float32
            attn_score: (batch_size, trg_seq_len, src_seq_len) of float32
        """
        # assert encoder_state is not None

        result, self.attn_scores = self.attention(
            x, target_embedding=None, encoder_outs=encoder_state, src_lengths=lengths)
        return result


class AddOp(BlockCombineNodeOp):
    def forward(self, in1, in2, lengths=None, encoder_state=None):
        return in1 + in2
