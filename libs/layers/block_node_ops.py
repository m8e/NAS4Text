#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Block node ops of block child network."""

# TODO: Add 1x1 conv in all ops to allow shape mismatch.

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
        self.controller = kwargs.pop('controller', None)
        self.index = kwargs.pop('index', None)
        self.input_index = kwargs.pop('input_index', None)

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def supported_ops():
        return {
            'LSTM': LSTMOp,
            'CNN': ConvolutionOp,
            'SelfAttention': SelfAttentionOp,
            'FFN': FFNOp,
            'PFFN': PFFNOp,
            'Identity': IdentityOp,
            'EncoderAttention': EncoderAttentionOp,
            'Zero': ZeroOp,
        }

    @classmethod
    def create(cls, op_code, op_args, input_shape, in_encoder=True, hparams=None, **kwargs):
        """

        Args:
            op_code: String op code.
            op_args:
            input_shape:
            in_encoder:
            hparams:
            **kwargs:

        Returns:

        """

        supported_ops = cls.supported_ops()
        op_type = supported_ops.get(op_code, None)
        if op_type is None:
            raise NotImplementedError('The op {!r} is not implemented now'.format(op_code))

        # Add some special cases here.

        return op_type(op_args, input_shape, hparams=hparams, in_encoder=in_encoder, **kwargs)


class BlockCombineNodeOp(nn.Module):
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__()
        self.op_args = op_args
        self.input_shape = input_shape
        self.hparams = kwargs.pop('hparams', None)
        self.in_encoder = kwargs.pop('in_encoder', True)
        self.controller = kwargs.pop('controller', None)
        self.index = kwargs.pop('index', None)

    def forward(self, in1, in2, lengths=None, encoder_state=None):
        raise NotImplementedError()

    @staticmethod
    def supported_ops():
        return {
            'Add': AddOp,
            'Concat': ConcatOp,
        }

    @classmethod
    def create(cls, op_code, op_args, input_shape, in_encoder=True, hparams=None, **kwargs):
        """

        Args:
            op_code: String op code.
            op_args:
            input_shape:
            in_encoder:
            hparams:
            **kwargs:

        Returns:

        """
        supported_ops = cls.supported_ops()
        op_type = supported_ops.get(op_code, None)
        if op_type is None:
            raise NotImplementedError('The combine op {!r} is not implemented now'.format(op_code))

        # Add some special cases here.

        return op_type(op_args, input_shape, hparams=hparams, **kwargs)


class ZeroOp(BlockNodeOp):
    """
    op_args: []
    """
    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return x.mul(0.0)


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
    op_args: [preprocessors = "", postprocessors = "", dim: index = None]
    """

    # TODO: Other args?

    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)
        input_size = input_shape[-1]

        preprocessors = _get_op_arg(self, 0, "")
        postprocessors = _get_op_arg(self, 1, "")
        d_ff = _get_op_arg(self, 2, self.hparams.attn_d_ff, space=ss.AttentionSpaces[self.hparams.attn_space].FFNSize)

        self.pffn = PositionwiseFeedForward(
            input_size, d_ff,
            dropout=self.hparams.ffn_dropout,
            hparams=self.hparams,
            linear_bias=self.hparams.attn_linear_bias,
        )
        push_prepostprocessors(self.pffn, preprocessors, postprocessors, input_shape, input_shape)

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.pffn(x, **kwargs)


class LSTMOp(BlockNodeOp):
    """
    op_args: [hidden_size(?), reversed: bool = False, preprocessors = "", postprocessors = ""]

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
        preprocessors = _get_op_arg(self, 2, '')
        postprocessors = _get_op_arg(self, 3, '')

        self.lstm = LSTMLayer(
            hparams=self.hparams,
            input_size=input_size,
            hidden_size=input_size,
            num_layers=1,
            bias=True,
            batch_first=not self.hparams.time_first,
            dropout=self.hparams.dropout,
            bidirectional=False,
            in_encoder=self.in_encoder,
            reversed=reversed_,
        )
        push_prepostprocessors(self.lstm, preprocessors, postprocessors, input_shape, input_shape)

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        encoder_state = kwargs.pop('encoder_state_mean', None)
        return self.lstm(x, lengths=lengths, encoder_state=encoder_state, **kwargs)


class ConvolutionOp(BlockNodeOp):
    """
    op_args: [out_channels(?), kernel_size: index = 3, stride: index = 1, groups: index = 1,
              preprocessors = "", postprocessors = ""]

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
        preprocessors = _get_op_arg(self, 4, '')
        postprocessors = _get_op_arg(self, 5, '')

        if self.in_encoder:
            conv_type = EncoderConvLayer
        else:
            conv_type = DecoderConvLayer

        self.conv = conv_type(self.hparams, in_channels=input_size, out_channels=input_size,
                              kernel_size=kernel_size, stride=stride, groups=groups)
        push_prepostprocessors(self.conv, preprocessors, postprocessors, input_shape, input_shape)

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        return self.conv(x, lengths=lengths, encoder_state=encoder_state, **kwargs)


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
            subsequent_mask=not self.in_encoder,
            ppp_args=[preprocessors, postprocessors],
        )

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        # [NOTE]: Override 'src_lengths' in kwargs with self lengths
        kwargs['src_lengths'] = lengths
        return self.attention(x, x, x, **kwargs)


class EncoderAttentionOp(BlockNodeOp):
    """
    op_args: [num_heads: index = 8, ..., preprocessors = "", postprocessors = ""]
    """

    # FIXME: The decoder must contains at least one encoder attention op,
    # or the encoder is not connected to the network,
    # and will raise the error of "grad is None, does not have data", etc.
    def __init__(self, op_args, input_shape, **kwargs):
        super().__init__(op_args, input_shape, **kwargs)

        if self.in_encoder:
            raise RuntimeError('Encoder attention only available in decoder')

        space = ss.AttentionSpaces[self.hparams.attn_space]
        h = _get_op_arg(self, 0, 8, space=space.NumHeads)
        preprocessors = _get_op_arg(self, 1, "")
        postprocessors = _get_op_arg(self, 2, "")

        self.attention = MultiHeadAttention(
            h,
            d_model=self.hparams.trg_embedding_size,
            d_q=input_shape[2], d_kv=self.hparams.src_embedding_size,
            dropout=self.hparams.attention_dropout, in_encoder=self.in_encoder, hparams=self.hparams,
            linear_bias=self.hparams.attn_linear_bias, subsequent_mask=False, attn_mean=True,
            ppp_args=[preprocessors, postprocessors],
        )

        self.attn_scores = None

    def forward(self, x, lengths=None, encoder_state=None, **kwargs):
        """
        Args:
            x: (batch_size, trg_seq_len, conv_channels) of float32
                If time_first: (trg_seq_len, batch_size, src_emb_size) of float32
            encoder_state (dict):
                'x': output, (batch_size, src_seq_len, src_emb_size) of float32
                    If time_first: (src_seq_len, batch_size, src_emb_size) of float32
                'y': output add source embedding, same shape as output
                    If time_first: same shape as output
                'src_mask':
            lengths: (batch_size,) of long

        Returns:
            output: (batch_size, trg_seq_len, conv_channels) of float32
                If time_first: (trg_seq_len, batch_size, src_emb_size) of float32
            attn_score: (batch_size, trg_seq_len, src_seq_len) of float32
                If time_first: (trg_seq_len, batch_size, src_emb_size) of float32
        """
        # assert encoder_state is not None

        # [NOTE]: Use 'src_lengths' in kwargs, does not use self 'lengths'
        # [NOTE]: Override 'mask' in kwargs with 'src_mask' of encoder state
        kwargs['mask'] = encoder_state['src_mask']
        result = self.attention(
            x, encoder_state['x'], encoder_state['y'], **kwargs,
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

        self.reduce_op = nn.Conv1d(2 * input_size, input_size, kernel_size=1, padding=0)

    def forward(self, in1, in2, lengths=None, encoder_state=None):
        return self.reduce_op(th.cat([in1, in2], dim=-1).transpose(1, 2)).transpose(1, 2)
