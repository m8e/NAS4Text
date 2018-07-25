#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Block node ops of block child network."""

import torch as th
import torch.nn as nn

from .common import *
from .lstm import LSTMLayer
from .cnn import EncoderConvLayer, DecoderConvLayer
from .multi_head_attention import MultiHeadAttention, PositionwiseFeedForward
from .ppp import push_prepostprocessors
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
        if op_code == ss.CellSpace.CellOps['LSTM']:
            op_type = LSTMOp
        elif op_code == ss.CellSpace.CellOps['CNN']:
            op_type = ConvolutionOp
        elif op_code == ss.CellSpace.CellOps['SelfAttention']:
            op_type = SelfAttentionOp
        elif op_code == ss.CellSpace.CellOps['FFN']:
            op_type = FFNOp
        elif op_code == ss.CellSpace.CellOps['PFFN']:
            op_type = PFFNOp
        elif op_code == ss.CellSpace.CellOps['Identity']:
            op_type = IdentityOp
        elif op_code == ss.CellSpace.CellOps['GroupedLSTM']:
            raise NotImplementedError()
        elif op_code == ss.CellSpace.CellOps['EncoderAttention']:
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
        if op_code == ss.CellSpace.CombineOps['Add']:
            op_type = AddOp
        elif op_code == ss.CellSpace.CombineOps['Concat']:
            op_type = ConcatOp
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

        space = ss.CellSpace.Activations
        activation = _get_op_arg(self, 0, space['identity'])
        if isinstance(activation, str):
            activation = space[activation]
        if activation == space['identity']:
            self.activation = Identity()
        elif activation == space['tanh']:
            self.activation = nn.Tanh()
        elif activation == space['relu']:
            self.activation = nn.ReLU()
        elif activation == space['sigmoid']:
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
    op_args: [..., preprocessors = "", postprocessors = ""]
    """

    # TODO: Other args?

    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)
        input_size = input_shape[-1]

        preprocessors = _get_op_arg(self, 0, "")
        postprocessors = _get_op_arg(self, 1, "")

        self.pffn = PositionwiseFeedForward(
            input_size, self.hparams.attn_d_ff,
            dropout=self.hparams.attention_dropout,
            hparams=self.hparams,
            linear_bias=self.hparams.attn_linear_bias,
        )
        push_prepostprocessors(self.pffn, preprocessors, postprocessors, input_shape, input_shape)

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.pffn(x)


class LSTMOp(BlockNodeOp):
    """
    op_args: [hidden_size(?), reversed: bool = False]

    [NOTE]: Unlike default network:
        The LSTM op is left-to-right (bidirectional = False).
        The LSTM op only contains 1 layer (num_layers = 1).
        The hidden size is same as input size now.
        The parameter of encoder state mean (used to initialize the hidden states)
            is passed as "encoder_state_mean", not "encoder_state".
    """

    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)
        input_size = input_shape[-1]

        reversed_ = _get_op_arg(self, 1, False)

        if reversed_ and not self.in_encoder:
            raise RuntimeError('R2L LSTM layer only available in encoder')

        self.lstm = LSTMLayer(
            hparams=self.hparams,
            input_size=input_size,
            hidden_size=input_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=self.hparams.dropout,
            bidirectional=False,
            in_encoder=self.in_encoder,
            reversed=reversed_,
        ).simplify()

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.lstm(x, lengths=lengths, encoder_state=kwargs.get('encoder_state_mean', None))


class ConvolutionOp(BlockNodeOp):
    """
    op_args: [out_channels(?), kernel_size: index = 3, stride: index = 1, groups: index = 1]

    [NOTE]: Unlike default network:
        The hidden size is same as input size now.
    """
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)

        input_size = input_shape[-1]

        space = ss.ConvolutionalSpaces[self.hparams.conv_space]
        kernel_size = _get_op_arg(self, 1, 3, space=space.KernelSizes)
        stride = _get_op_arg(self, 2, 1, space=space.Strides)
        groups = _get_op_arg(self, 3, 1, space=space.Groups)

        if self.in_encoder:
            conv_type = EncoderConvLayer
        else:
            conv_type = DecoderConvLayer

        self.conv = conv_type(self.hparams, in_channels=input_size, out_channels=input_size,
                              kernel_size=kernel_size, stride=stride, groups=groups).simplify()

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.conv(x, lengths=lengths, encoder_state=encoder_state)


class SelfAttentionOp(BlockNodeOp):
    """
    op_args: [num_heads: index = 8, ..., preprocessors = "", postprocessors = ""]
    """

    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)
        input_size = input_shape[-1]

        space = ss.AttentionSpaces[self.hparams.attn_space]
        h = _get_op_arg(self, 0, 8, space=space.NumHeads)
        preprocessors = _get_op_arg(self, 1, "")
        postprocessors = _get_op_arg(self, 2, "")

        self.attention = MultiHeadAttention(
            h, input_size,
            hparams=self.hparams,
            in_encoder=self.in_encoder,
            linear_bias=self.hparams.attn_linear_bias,
            dropout=self.hparams.attention_dropout,
            subsequent_mask=True,
        )
        push_prepostprocessors(self.attention, preprocessors, postprocessors, input_shape, input_shape)

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.attention(x, x, x, lengths)


class EncoderAttentionOp(BlockNodeOp):
    """
    op_args: [num_heads: index = 8, ..., preprocessors = "", postprocessors = ""]
    """

    # FIXME: The decoder must contains at least one encoder attention op,
    # or the encoder is not connected to the network,
    # and will raise the error of "grad is None, does not have data", etc.
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)

        space = ss.AttentionSpaces[self.hparams.attn_space]
        h = _get_op_arg(self, 0, 8, space=space.NumHeads)
        preprocessors = _get_op_arg(self, 1, "")
        postprocessors = _get_op_arg(self, 2, "")

        self.attention = MultiHeadAttention(
            h,
            d_model=self.hparams.trg_embedding_size,
            d_q=input_shape[2], d_kv=self.hparams.src_embedding_size,
            dropout=self.hparams.attention_dropout, in_encoder=False, hparams=self.hparams,
            linear_bias=self.hparams.attn_linear_bias, subsequent_mask=False, attn_mean=True,
        )
        push_prepostprocessors(self.attention, preprocessors, postprocessors, input_shape, input_shape)

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

        result = self.attention(
            x, encoder_state[0], encoder_state[1], src_lengths=kwargs.get('src_lengths', None),
            target_embedding=kwargs.get('target_embedding', None),
        )
        self.attn_scores = self.attention.attn

        return result


class AddOp(BlockCombineNodeOp):
    """
    op_args: []
    """
    def forward(self, in1, in2, lengths=None, encoder_state=None):
        return in1 + in2


class ConcatOp(BlockCombineNodeOp):
    """
    op_args: []
    """

    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)
        input_size = input_shape[-1]

        self.linear = Linear(2 * input_size, input_size, bias=True, dropout=self.hparams.dropout, hparams=self.hparams)

    def forward(self, in1, in2, lengths=None, encoder_state=None):
        return self.linear(th.cat([in1, in2], dim=-1))