#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .child_net_base import ChildNetBase, EncDecChildNet, ChildDecoderBase
from ..utils.data_processing import LanguagePairDataset
from ..tasks import get_task
from ..layers.common import *
from ..layers.build_block import build_block
from ..layers.grad_multiply import GradMultiply

__author__ = 'fyabc'


class BlockChildEncoder(nn.Module):
    def __init__(self, code, hparams):
        super().__init__()

        self.code = code
        self.hparams = hparams
        self.task = get_task(hparams.task)

        # Encoder input shape (after embedding).
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.src_embedding_size])

        # Embeddings.
        self.embed_tokens = Embedding(self.task.SourceVocabSize, hparams.src_embedding_size, self.task.PAD_ID,
                                      hparams=hparams)
        self.embed_positions = PositionalEmbedding(
            hparams.max_src_positions,
            hparams.src_embedding_size,
            self.task.PAD_ID,
            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE,
            hparams=hparams,
        )

        # The main encoder network.
        self.layers = nn.ModuleList()
        input_shape = self.input_shape
        # self.fc1 = Linear(input_shape[2], 128)
        # input_shape = th.Size([1, 1, 128])
        for i, layer_code in enumerate(code):
            layer, output_shape = build_block(layer_code, input_shape, self.hparams, in_encoder=True)
            self.layers.append(layer)

            input_shape = output_shape

        if hparams.enc_output_fc or input_shape[2] != hparams.src_embedding_size:
            self.fc2 = Linear(input_shape[2], hparams.src_embedding_size, hparams=hparams)
        else:
            self.fc2 = None

        # Encoder output shape
        self.output_shape = th.Size([input_shape[0], input_shape[1], hparams.src_embedding_size])

        self.out_norm = LayerNorm(self.output_shape[2])

    @property
    def num_layers(self):
        return len(self.layers)

    def forward(self, src_tokens, src_lengths=None):
        """

                Args:
                    src_tokens: (batch_size, src_seq_len) of int32
                    src_lengths: (batch_size,) of long

                Returns:
                    Output: (batch_size, src_seq_len, src_emb_size) of float32
                    Output with embedding: (batch_size, src_seq_len, src_emb_size) of float32
                """
        x = src_tokens
        logging.debug('Encoder input shape: {}'.format(list(x.shape)))

        x = self.embed_tokens(x) + self.embed_positions(x)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        source_embedding = x

        # x = self.fc1(x)

        logging.debug('Encoder input shape after embedding: {}'.format(list(x.shape)))
        input_list = [None, x]
        for i in range(self.num_layers):
            layer = self.get_layer(i)
            output = layer(input_list[-1], input_list[-2], lengths=src_lengths)
            input_list.append(output)

            logging.debug('Encoder layer {} output shape: {}'.format(i, list(output.shape)))

        # Output normalization
        x = self.out_norm(x)

        # project back to size of embedding
        if self.fc2 is not None:
            x = self.fc2(x)

        if self.hparams.apply_grad_mul:
            # scale gradients (this only affects backward, not forward)
            x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        if self.hparams.connect_src_emb:
            # add output to input embedding for attention
            y = (x + source_embedding) * math.sqrt(0.5)
        else:
            y = x

        logging.debug('Encoder output shape: {} & {}'.format(list(x.shape), list(y.shape)))
        return x, y

    def upgrade_state_dict(self, state_dict):
        return state_dict

    def max_positions(self):
        return self.embed_positions.max_positions()

    def get_layer(self, i):
        return getattr(self, 'layer_{}'.format(i))

    def get_layers(self):
        return [self.get_layer(i) for i in range(self.num_layers)]


class BlockChildDecoder(ChildDecoderBase):
    def __init__(self, code, hparams, **kwargs):
        super().__init__(code, hparams, **kwargs)

        # Decoder input shape (after embedding)
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.trg_embedding_size])

        self._build_embedding(kwargs.pop('src_embedding'))

        self.layers = nn.ModuleList()

        input_shape = self.input_shape
        for i, layer_code in enumerate(code):
            # TODO: Add attention shape info in ``build_block``.
            layer, output_shape = build_block(layer_code, input_shape, self.hparams, in_encoder=False)
            self.layers.append(layer)

            input_shape = output_shape

        # Decoder output shape (before softmax)
        self.output_shape = input_shape

        if hparams.dec_output_fc or self.output_shape[2] != hparams.decoder_out_embedding_size:
            self.fc2 = Linear(self.output_shape[2], hparams.decoder_out_embedding_size, hparams=hparams)
        else:
            self.fc2 = None
        self.out_norm = LayerNorm(self.output_shape[2])

        self._build_fc_last()

    @property
    def num_layers(self):
        return len(self.layers)

    def forward(self, encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state=None):
        """

        Args:
            encoder_out (tuple):
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

        # TODO


@ChildNetBase.register_child_net
class BlockChildNet(EncDecChildNet):
    def __init__(self, net_code, hparams):
        super().__init__(net_code, hparams)

        self.task = get_task(hparams.task)

        self.encoder = BlockChildEncoder(net_code[0], hparams)
        self.decoder = BlockChildDecoder(net_code[1], hparams, src_embedding=self.encoder.embed_tokens)
