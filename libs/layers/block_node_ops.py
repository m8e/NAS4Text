#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Block node ops of block child network."""

import torch as th
import torch.nn as nn

from .common import *
from .cnn import EncoderConvLayer
from .multi_head_attention import EncDecAttention, MHAttentionWrapper, PositionwiseFeedForward
from ..utils import search_space as ss

__author__ = 'fyabc'


def _get_op_arg(self, i, default=None, space=None):
    try:
        result = self.op_args[i]
        if space is not None:
            index = result
            return space[index]
        else:
            return result
    except IndexError:
        return default


class BlockNodeOp(nn.Module):
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__()
        self.op_args = op_args
        self.input_shape = input_shape
        self.hparams = kwargs.pop('hparams', None)
        self.in_encoder = kwargs.pop('in_encoder', True)

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def create(op_code, op_args, input_shape, in_encoder=True, hparams=None):
        if op_code == ss.CellSpace.LSTM:
            raise NotImplementedError()
        elif op_code == ss.CellSpace.R2L_LSTM:
            raise NotImplementedError()
        elif op_code == ss.CellSpace.CNN:
            op_type = ConvolutionOp
        elif op_code == ss.CellSpace.SelfAttention:
            op_type = SelfAttentionOp
        elif op_code == ss.CellSpace.GroupedCNN:
            raise NotImplementedError()
        elif op_code == ss.CellSpace.FFN:
            op_type = FFNOp
        elif op_code == ss.CellSpace.PFFN:
            op_type = PFFNOp
        elif op_code == ss.CellSpace.Identity:
            op_type = IdentityOp
        elif op_code == ss.CellSpace.GroupedLSTM:
            raise NotImplementedError()
        elif op_code == ss.CellSpace.EncoderAttention:
            if in_encoder:
                raise RuntimeError('Encoder attention only available in decoder')
            op_type = EncoderAttentionOp
        else:
            raise RuntimeError('Unknown op code {}'.format(op_code))
        return op_type(op_args, input_shape, hparams=hparams, in_encoder=in_encoder)


class BlockCombineNodeOp(nn.Module):
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__()
        self.op_args = op_args
        self.input_shape = input_shape
        self.hparams = kwargs.pop('hparams', None)
        self.in_encoder = kwargs.pop('in_encoder', True)

    def forward(self, in1, in2, lengths=None, encoder_state=None):
        raise NotImplementedError()

    @staticmethod
    def create(op_code, op_args, input_shape, in_encoder=True, hparams=None):
        if op_code == ss.CellSpace.Add:
            op_type = AddOp
        elif op_code == ss.CellSpace.Concat:
            raise NotImplementedError()
        else:
            raise RuntimeError('Unknown combine op code {}'.format(op_code))
        return op_type(op_args, input_shape, hparams=hparams)


class IdentityOp(BlockNodeOp):
    """
    op_args: []
    """
    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return x


class FFNOp(BlockNodeOp):
    """
    op_args: [activation: index or str = identity, have_bias: bool = True, output_size: int = None]
    """
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)
        input_size = input_shape[-1]

        space = ss.CellSpace.ActivationSpace
        activation = _get_op_arg(self, 0, space.identity)
        activation = ss.space_str2int(space, activation)
        if activation == space.identity:
            self.activation = Identity()
        elif activation == space.tanh:
            self.activation = nn.Tanh()
        elif activation == space.relu:
            self.activation = nn.ReLU()
        elif activation == space.sigmoid:
            self.activation = nn.Sigmoid()
        else:
            raise RuntimeError('Unknown activation type {}'.format(activation))
        bias = _get_op_arg(self, 1, True)
        output_size = _get_op_arg(self, 2, input_size)

        self.linear = Linear(input_size, output_size, bias=bias, hparams=self.hparams)

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.activation(self.linear(x))


class PFFNOp(BlockNodeOp):
    """
    op_args: []
    """

    # TODO: Other args?

    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)
        input_size = input_shape[-1]

        self.pffn = PositionwiseFeedForward(
            input_size, self.hparams.attn_d_ff,
            dropout=self.hparams.attention_dropout,
            hparams=self.hparams,
            linear_bias=self.hparams.attn_linear_bias,
        )

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.pffn(x)


class ConvolutionOp(BlockNodeOp):
    """
    op_args: [out_channels(?), kernel_size: index = 3, stride: index = 1]
    """
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)

        input_size = input_shape[-1]

        space = ss.ConvolutionalSpaces[self.hparams.conv_space]
        kernel_size = _get_op_arg(self, 1, 3, space=space.KernelSizes)
        stride = _get_op_arg(self, 2, 1, space=space.Strides)

        self.conv = EncoderConvLayer(self.hparams, in_channels=input_size, out_channels=input_size,
                                     kernel_size=kernel_size, stride=stride).simplify()

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.conv(x, lengths=lengths, encoder_state=encoder_state)


class SelfAttentionOp(BlockNodeOp):
    """
    op_args: [num_heads: index = 8]
    """

    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)
        input_size = input_shape[-1]

        space = ss.AttentionSpaces[self.hparams.attn_space]
        h = _get_op_arg(self, 0, 8, space=space.NumHeads)

        self.attention = MHAttentionWrapper(
            h, input_size,
            hparams=self.hparams,
            in_encoder=self.in_encoder,
            linear_bias=self.hparams.attn_linear_bias,
            dropout=self.hparams.attention_dropout,
        )

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.attention(x, lengths=lengths, encoder_state=encoder_state)


class EncoderAttentionOp(BlockNodeOp):
    """
    op_args: [num_heads: index = 8]
    """

    # FIXME: The decoder must contains at least one encoder attention op,
    # or the encoder is not connected to the network,
    # and will raise the error of "grad is None, does not have data", etc.
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)

        space = ss.AttentionSpaces[self.hparams.attn_space]
        h = _get_op_arg(self, 0, 8, space=space.NumHeads)

        self.attention = EncDecAttention(
            h,
            input_shape[2],
            self.hparams.trg_embedding_size, self.hparams.src_embedding_size,
            dropout=self.hparams.dropout, in_encoder=False, hparams=self.hparams,
            linear_bias=self.hparams.attn_linear_bias,
        )

        self.attn_scores = None

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
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
            x, target_embedding=None, encoder_outs=encoder_state, src_lengths=kwargs.get('src_lengths', None))
        return result


class AddOp(BlockCombineNodeOp):
    """
    op_args: []
    """
    def forward(self, in1, in2, lengths=None, encoder_state=None):
        return in1 + in2
