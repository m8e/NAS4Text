#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Try to build network from code (Neural Architecture Search results).

Network architecture:

Layer format: callable (nn.Module instance recommended)
Inputs:
    x:
    lengths:
    **kwargs:
Output:
    out:
"""

# TODO: Merge similar code in encoder and decoder.

import logging
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .child_net_base import ChildNetBase, EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase
from ..utils.search_space import LayerTypes
from ..utils import common
from ..layers.common import *
from ..layers.lstm import build_lstm, LSTMLayer
from ..layers.cnn import build_cnn
from ..layers.attention import build_attention
from ..layers.multi_head_attention import MultiHeadAttention
from ..layers.grad_multiply import GradMultiply

__author__ = 'fyabc'


def _code2layer(layer_code, input_shape, hparams, in_encoder=True):
    layer_type = layer_code[0]

    if layer_type == LayerTypes.LSTM:
        return build_lstm(layer_code, input_shape, hparams, in_encoder)
    elif layer_type == LayerTypes.Convolutional:
        return build_cnn(layer_code, input_shape, hparams, in_encoder)
    elif layer_type == LayerTypes.Attention:
        return build_attention(layer_code, input_shape, hparams, in_encoder)
    else:
        raise ValueError('Unknown layer type {}'.format(layer_type))


class ChildEncoder(ChildEncoderBase):
    def __init__(self, code, hparams, embed_tokens):
        super().__init__(code, hparams)

        # Value from decoder
        self.num_attention_layers = None

        # Encoder input shape (after embedding).
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.src_embedding_size])

        # Embeddings.
        self._build_embedding(embed_tokens)

        # TODO: Need fc1 to project src_emb_size to in_channels here?

        # The main encoder network.
        # [NOTE]: If we put layers into `self._net`, this member will not update when duplicating the model,
        # so `self._net` will point to the modules of other model replicas (on another GPUs), which is wrong.
        self.num_layers = 0
        input_shape = self.input_shape
        # self.fc1 = Linear(input_shape[2], 128)
        # input_shape = th.Size([1, 1, 128])
        for i, layer_code in enumerate(code):
            layer, output_shape = _code2layer(layer_code, input_shape, self.hparams, in_encoder=True)
            setattr(self, 'layer_{}'.format(i), layer)

            input_shape = output_shape
            self.num_layers += 1

        self._init_post(input_shape)

    def forward(self, src_tokens, src_lengths=None):
        """

        Args:
            src_tokens: (batch_size, src_seq_len) of int32
            src_lengths: (batch_size,) of long

        Returns:
            Output: (batch_size, src_seq_len, src_emb_size) of float32
            Output with embedding: (batch_size, src_seq_len, src_emb_size) of float32
        """

        x, src_mask, source_embedding = self._fwd_pre(src_tokens, src_lengths)

        for i in range(self.num_layers):
            layer = self.get_layer(i)

            x = layer(x, src_lengths, mask=src_mask)

            logging.debug('Encoder layer {} output shape: {}'.format(i, list(x.shape)))

        return self._fwd_post(x, src_mask, source_embedding)

    def reorder_encoder_out(self, encoder_out, new_order):
        # TODO: Implement this method.
        return encoder_out

    def get_layer(self, i):
        return getattr(self, 'layer_{}'.format(i))

    def get_layers(self):
        return [self.get_layer(i) for i in range(self.num_layers)]


class ChildDecoder(ChildIncrementalDecoderBase):
    def __init__(self, code, hparams, embed_tokens):
        super().__init__(code, hparams)

        # Decoder input shape (after embedding)
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.trg_embedding_size])

        self._build_embedding(embed_tokens)

        # The main decoder network.
        self.num_layers = 0

        input_shape = self.input_shape
        for i, layer_code in enumerate(code):
            layer, output_shape = _code2layer(layer_code, input_shape, self.hparams, in_encoder=False)
            setattr(self, 'layer_{}'.format(i), layer)

            if getattr(layer, 'encdec_attention', None) is None:
                assert hparams.enc_dec_attn_type == 'dot_product', 'Only support multi-head attention now'
                attention = MultiHeadAttention(
                    8, d_model=hparams.trg_embedding_size, d_q=output_shape[2], d_kv=hparams.src_embedding_size,
                    dropout=hparams.attention_dropout, in_encoder=False, hparams=hparams,
                    linear_bias=hparams.attn_linear_bias, subsequent_mask=False, attn_mean=True,
                    ppp_args=['', 'dan'],
                )
            else:
                attention = None
            setattr(self, 'attention_{}'.format(i), attention)

            input_shape = output_shape
            self.num_layers += 1

        self._init_post(input_shape)

    def forward(self, encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state=None):
        """

        Args:
            encoder_out (dict):
                output: (batch_size, src_seq_len, src_emb_size) of float32
                output add source embedding: same shape as output
            src_lengths: (batch_size,) of long
            trg_tokens: (batch_size, trg_seq_len) of int32
            trg_lengths: (batch_size,) of long
            incremental_state: Incremental states for decoding. TODO

        Returns:
            Output: (batch_size, trg_seq_len, trg_vocab_size) of float32
            Attention scores: (batch_size, trg_seq_len, src_seq_len) of float32
        """

        x, encoder_out, trg_mask, target_embedding, encoder_state_mean = self._fwd_pre(
            encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state
        )

        avg_attn_scores = None
        num_attn_layers = self.num_layers  # TODO: Explain why include layers without attention (None)?
        for i in range(self.num_layers):
            layer = self.get_layer(i)
            attention = self.get_attention(i)

            x = layer(
                x, trg_lengths,
                encoder_state=encoder_state_mean, encoder_out=encoder_out,
                target_embedding=target_embedding if self.hparams.connect_trg_emb else None,
                src_lengths=src_lengths, mask=trg_mask,
            )

            if attention is None:
                attn_scores = layer.attn_scores
            else:
                x, attn_scores = attention(
                    x, encoder_out['x'], encoder_out['y'],
                    target_embedding=target_embedding if self.hparams.connect_trg_emb else None,
                    src_lengths=src_lengths, mask=encoder_out['src_mask'])

            attn_scores = attn_scores / num_attn_layers
            if avg_attn_scores is None:
                avg_attn_scores = attn_scores
            else:
                avg_attn_scores.add_(attn_scores)

            logging.debug('Decoder layer {} output shape: {}'.format(i, list(x.shape)))

        return self._fwd_post(x, avg_attn_scores)

    def _contains_lstm(self):
        return any(isinstance(l, LSTMLayer) for l in self.get_layers())

    @property
    def num_attention_layers(self):
        return sum(a is not None for a in self.get_attentions())

    def get_layer(self, i):
        return getattr(self, 'layer_{}'.format(i))

    def get_layers(self):
        return [self.get_layer(i) for i in range(self.num_layers)]

    def get_attention(self, i):
        return getattr(self, 'attention_{}'.format(i))

    def get_attentions(self):
        return [self.get_attention(i) for i in range(self.num_layers)]


@ChildNetBase.register_child_net
class ChildNet(EncDecChildNet):
    def __init__(self, net_code, hparams):
        super().__init__(net_code, hparams)

        src_embed_tokens, trg_embed_tokens = self._build_embed_tokens()

        self.encoder = ChildEncoder(net_code[0], hparams, src_embed_tokens)
        self.decoder = ChildDecoder(net_code[1], hparams, trg_embed_tokens)
        self.encoder.num_attention_layers = self.decoder.num_attention_layers
