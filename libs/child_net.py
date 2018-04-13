#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Try to build network from code (Neural Architecture Search results).

Network architecture:

Layer format: callable (nn.Module instance recommended)
Inputs:
    x:
    mask:
Output:
    out:
"""

# TODO: Merge similar code in encoder and decoder.

import logging
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .tasks import get_task
from .utils.data_processing import LanguagePairDataset
from .layers.common import Linear, Embedding, PositionalEmbedding
from .layers.net_code import NetCodeEnum
from .layers.lstm import build_lstm
from .layers.cnn import build_cnn
from .layers.attention import build_attention

__author__ = 'fyabc'


def _code2layer(layer_code, input_shape, hparams, in_encoder=True):
    layer_type = layer_code[0]

    if layer_type == NetCodeEnum.LSTM:
        return build_lstm(layer_code, input_shape, hparams, in_encoder)
    elif layer_type == NetCodeEnum.Convolutional:
        return build_cnn(layer_code, input_shape, hparams, in_encoder)
    elif layer_type == NetCodeEnum.Attention:
        return build_attention(layer_code, input_shape, hparams, in_encoder)
    else:
        raise ValueError('Unknown layer type {}'.format(layer_type))


class ChildEncoder(nn.Module):
    def __init__(self, code, hparams):
        super().__init__()

        self.code = code
        self.hparams = hparams
        self.task = get_task(hparams.task)

        # Encoder input shape (after embedding).
        # [NOTE]: The shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([hparams.batch_size, 1, hparams.src_embedding_size])

        self.embed_tokens = Embedding(self.task.SourceVocabSize, hparams.src_embedding_size, self.task.PAD_ID)
        self.embed_positions = PositionalEmbedding(
            hparams.src_seq_length,
            hparams.src_embedding_size,
            self.task.PAD_ID,
            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE,
        )

        # The main encoder network.
        self._net = []
        self._projections = []

        input_shape = self.input_shape
        for i, layer_code in enumerate(code):
            layer, output_shape = _code2layer(layer_code, input_shape, self.hparams, in_encoder=True)
            self._net.append(layer)
            setattr(self, 'layer_{}'.format(i), layer)

            projection = Linear(input_shape[2], output_shape[2]) if input_shape != output_shape else None
            self._projections.append(projection)
            setattr(self, 'projection_{}'.format(i), projection)
            input_shape = output_shape

        # Encoder output shape
        self.output_shape = input_shape

    def forward(self, src_tokens, src_mask=None):
        """

        Args:
            src_tokens: (batch_size, src_seq_len) of int32
            src_mask: (batch_size, src_seq_len) of byte

        Returns:
            (batch_size, src_seq_len, encoder_out_channels) of float32
        """
        x = src_tokens
        logging.debug('Encoder input shape: {}'.format(list(x.shape)))

        x = self.embed_tokens(x) + self.embed_positions(x)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        source_embedding = x

        logging.debug('Encoder input shape after embedding: {}'.format(list(x.shape)))
        for i, (layer, projection) in enumerate(zip(self._net, self._projections)):
            residual = x if projection is None else projection(x)

            x = layer(x, src_mask)

            # Residual connection.
            # If sequence length changed, add 1x1 convolution ([NOTE]: The layer must provide it).
            # We cannot determine sequence length when building the module, so test them here.
            if x.shape[1] != residual.shape[1]:
                residual = layer.residual_conv(residual.transpose(1, 2)).transpose(1, 2)
            x = (x + residual) * math.sqrt(0.5)

            logging.debug('Encoder layer {} output shape: {}'.format(i, list(x.shape)))

        return x


class ChildDecoder(nn.Module):
    def __init__(self, code, hparams, encoder_output_shape):
        super().__init__()

        self.code = code
        self.hparams = hparams
        self.task = get_task(hparams.task)

        # Encoder output shape
        self.encoder_output_shape = encoder_output_shape
        # Decoder input shape (after embedding)
        # [NOTE]: The shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([hparams.batch_size, 1, hparams.trg_embedding_size])

        self.embed_tokens = Embedding(self.task.TargetVocabSize, hparams.trg_embedding_size, self.task.PAD_ID)
        self.embed_positions = PositionalEmbedding(
            hparams.trg_seq_length,
            hparams.trg_embedding_size,
            self.task.PAD_ID,
            left_pad=LanguagePairDataset.LEFT_PAD_TARGET,
        )

        # The main decoder network.
        self._net = []
        self._projections = []

        input_shape = self.input_shape
        for i, layer_code in enumerate(code):
            layer, output_shape = _code2layer(layer_code, input_shape, self.hparams, in_encoder=False)
            self._net.append(layer)
            setattr(self, 'layer_{}'.format(i), layer)

            projection = Linear(input_shape[2], output_shape[2]) if input_shape != output_shape else None
            self._projections.append(projection)
            setattr(self, 'projection_{}'.format(i), projection)
            input_shape = output_shape

        # Decoder output shape (before softmax)
        self.output_shape = input_shape

        self.fc_last = Linear(self.output_shape[2], self.task.TargetVocabSize)

    def forward(self, encoder_out, src_mask, trg_tokens, trg_mask, incremental_state=None):
        """

        Args:
            encoder_out: (batch_size, src_seq_len, encoder_out_channels) of float32
            src_mask: (batch_size, src_seq_len) of byte
            trg_tokens: (batch_size, trg_seq_len) of int32
            trg_mask: (batch_size, trg_seq_len) of byte
            incremental_state: Incremental states for decoding. TODO

        Returns:

        """
        x = trg_tokens
        logging.debug('Decoder input shape: {}'.format(list(x.shape)))

        x = self._embed_tokens(x, incremental_state) + self.embed_positions(x, incremental_state)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        target_embedding = x

        logging.debug('Decoder input shape after embedding: {}'.format(list(x.shape)))
        for i, (layer, projection) in enumerate(zip(self._net, self._projections)):
            residual = x if projection is None else projection(x)

            x = layer(x, trg_mask)

            # TODO: Add attention layer here
            #   x, attn_scores = attention(x, target_embedding, encoder_outs)

            # Residual connection.
            # If sequence length changed, add 1x1 convolution ([NOTE]: The layer must provide it).
            # We cannot determine sequence length when building the module, so test them here.
            if x.shape[1] != residual.shape[1]:
                residual = layer.residual_conv(residual.transpose(1, 2)).transpose(1, 2)
            x = (x + residual) * math.sqrt(0.5)

            logging.debug('Decoder layer {} output shape: {}'.format(i, list(x.shape)))

        # Project back to size of vocabulary
        # TODO: fc2, dropout, fc3 in fairseq-py
        x = self.fc_last(x)

        logging.debug('Decoder output shape: {}'.format(list(x.shape)))
        return x

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # Keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    @staticmethod
    def get_normalized_probs(net_output, log_probs=False):
        logits = net_output
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)


class ChildNet(nn.Module):
    def __init__(self, net_code, hparams):
        super().__init__()

        self.net_code = net_code
        self.hparams = hparams
        self.task = get_task(hparams.task)

        self.encoder = ChildEncoder(net_code[0], hparams)
        self.decoder = ChildDecoder(net_code[1], hparams, encoder_output_shape=self.encoder.output_shape)

    def forward(self, src_tokens, src_mask, trg_tokens, trg_mask):
        """

        Args:
            src_tokens: (batch_size, src_seq_len) of int32
            src_mask: (batch_size, src_seq_len) of byte
            trg_tokens: (batch_size, trg_seq_len) of int32
            trg_mask: (batch_size, trg_seq_len) of byte

        Returns:
            (batch_size, seq_len, tgt_vocab_size) of float32
        """
        encoder_out = self.encoder(src_tokens, src_mask)
        decoder_out = self.decoder(encoder_out, src_mask, trg_tokens, trg_mask)

        return decoder_out

    def encode(self, src_tokens, src_mask):
        return self.encoder(src_tokens, src_mask)

    def decode(self, encoder_out, src_mask, trg_tokens, trg_mask):
        return self.decoder(encoder_out, src_mask, trg_tokens, trg_mask)

    def get_normalized_probs(self, net_output, log_probs=False):
        return self.decoder.get_normalized_probs(net_output, log_probs)
