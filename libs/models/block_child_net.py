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

        self.out_norm = LayerNorm(input_shape[2])

        if hparams.enc_output_fc or input_shape[2] != hparams.src_embedding_size:
            self.fc2 = Linear(input_shape[2], hparams.src_embedding_size, hparams=hparams)
        else:
            self.fc2 = None

        # Encoder output shape
        self.output_shape = th.Size([input_shape[0], input_shape[1], hparams.src_embedding_size])

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
            layer = self.layers[i]
            output = layer(input_list[-1], input_list[-2], lengths=src_lengths)
            input_list.append(output)

            logging.debug('Encoder layer {} output shape: {}'.format(i, list(output.shape)))
        x = input_list[-1]

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
            layer, output_shape = build_block(layer_code, input_shape, self.hparams, in_encoder=False)
            self.layers.append(layer)

            input_shape = output_shape

        # Decoder output shape (before softmax)
        self.output_shape = input_shape

        self.out_norm = LayerNorm(self.output_shape[2])
        if hparams.dec_output_fc or self.output_shape[2] != hparams.decoder_out_embedding_size:
            self.fc2 = Linear(self.output_shape[2], hparams.decoder_out_embedding_size, hparams=hparams)
        else:
            self.fc2 = None

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

        encoder_state_mean = self._get_encoder_state_mean(encoder_out, src_lengths)

        # split and (transpose) encoder outputs
        encoder_out = self._split_encoder_out(encoder_out, incremental_state)

        x = trg_tokens
        logging.debug('Decoder input shape: {}'.format(list(x.shape)))

        x = self._embed_tokens(x, incremental_state) + self.embed_positions(x, incremental_state)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        target_embedding = x

        logging.debug('Decoder input shape after embedding: {}'.format(list(x.shape)))
        input_list = [None, x]
        for i in range(self.num_layers):
            layer = self.layers[i]

            output = layer(
                input_list[-1], input_list[-2],
                lengths=trg_lengths, encoder_state=encoder_out, src_lengths=src_lengths,
                target_embedding=target_embedding, encoder_state_mean=encoder_state_mean,
            )
            input_list.append(output)

            logging.debug('Decoder layer {} output shape: {}'.format(i, list(x.shape)))
        x = input_list[-1]

        # Output normalization
        x = self.out_norm(x)

        # Project back to size of vocabulary
        if self.fc2 is not None:
            x = self.fc2(x)
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)

        x = self.fc_last(x)

        logging.debug('Decoder output shape: {} & None'.format(list(x.shape)))
        return x, None

    def _contains_lstm(self):
        return any(l.contains_lstm() for l in self.layers)


@ChildNetBase.register_child_net
class BlockChildNet(EncDecChildNet):
    def __init__(self, net_code, hparams):
        super().__init__(net_code, hparams)

        self.task = get_task(hparams.task)

        self.encoder = BlockChildEncoder(net_code[0], hparams)
        self.decoder = BlockChildDecoder(net_code[1], hparams, src_embedding=self.encoder.embed_tokens)
