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
from .layers.multi_head_attention import MultiHeadAttention

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
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.src_embedding_size])

        self.embed_tokens = Embedding(self.task.SourceVocabSize, hparams.src_embedding_size, self.task.PAD_ID)
        self.embed_positions = PositionalEmbedding(
            hparams.max_src_positions,
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

        self.fc2 = Linear(self.output_shape[2], hparams.src_embedding_size)

    def forward(self, src_tokens, src_lengths=None):
        """

        Args:
            src_tokens: (batch_size, src_seq_len) of int32
            src_lengths: (batch_size,) of long

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

            x = layer(x, src_lengths)

            # Residual connection.
            # If sequence length changed, add 1x1 convolution ([NOTE]: The layer must provide it).
            # We cannot determine sequence length when building the module, so test them here.
            if x.shape[1] != residual.shape[1]:
                residual = layer.residual_conv(residual.transpose(1, 2)).transpose(1, 2)
            x = (x + residual) * math.sqrt(0.5)

            logging.debug('Encoder layer {} output shape: {}'.format(i, list(x.shape)))

        # project back to size of embedding
        # TODO: Explain this, why need to fc1: emb -> conv & fc2: conv -> emb?
        x = self.fc2(x)

        # TODO: scale gradients (this only affects backward, not forward)

        # add output to input embedding for attention
        y = (x + source_embedding) * math.sqrt(0.5)

        return x, y

    def max_positions(self):
        return self.embed_positions.max_positions()


class EncDecAttention(nn.Module):
    def __init__(self, conv_channels, embed_dim, hparams):
        super().__init__()
        self.in_projection = Linear(conv_channels, embed_dim)
        # [NOTE]: h = 1 now; can modify it?
        self.multi_head = MultiHeadAttention(1, embed_dim, dropout=hparams.dropout, in_encoder=False)
        self.out_projection = Linear(conv_channels, embed_dim)

    def forward(self, x, target_embedding, encoder_outs, trg_lengths=None):
        """

        Args:
            x: (batch_size, trg_seq_len, conv_channels) of float32
            target_embedding: (batch_size, trg_seq_len, trg_emb_size) of float32
            encoder_outs (tuple):
                output: (batch_size, src_seq_len, encoder_out_channels) of float32
                output add source embedding: same shape as output
            trg_lengths: (batch_size, trg_seq_len) of long

        Returns:
            output: (batch_size, trg_seq_len, conv_channels) of float32
            attn_score
        """

        # TODO: Shape bug here, how to design the enc-dec attention?

        residual = x

        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)

        x = self.multi_head(x, encoder_outs[0], encoder_outs[1], trg_lengths)

        x = (self.out_projection(x) + residual) * math.sqrt(0.5)

        return x, self.multi_head.attn


class ChildDecoder(nn.Module):
    def __init__(self, code, hparams, encoder_output_shape):
        super().__init__()

        self.code = code
        self.hparams = hparams
        self.task = get_task(hparams.task)

        # Encoder output shape
        self.encoder_output_shape = encoder_output_shape
        # Decoder input shape (after embedding)
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.trg_embedding_size])

        self.embed_tokens = Embedding(self.task.TargetVocabSize, hparams.trg_embedding_size, self.task.PAD_ID)
        self.embed_positions = PositionalEmbedding(
            hparams.max_trg_positions,
            hparams.trg_embedding_size,
            self.task.PAD_ID,
            left_pad=LanguagePairDataset.LEFT_PAD_TARGET,
        )

        # The main decoder network.
        self._net = []
        self._projections = []
        self._attentions = []

        input_shape = self.input_shape
        for i, layer_code in enumerate(code):
            layer, output_shape = _code2layer(layer_code, input_shape, self.hparams, in_encoder=False)
            self._net.append(layer)
            setattr(self, 'layer_{}'.format(i), layer)

            projection = Linear(input_shape[2], output_shape[2]) if input_shape != output_shape else None
            self._projections.append(projection)
            setattr(self, 'projection_{}'.format(i), projection)

            attention = EncDecAttention(output_shape[2], hparams.trg_embedding_size, self.hparams)
            self._attentions.append(attention)
            setattr(self, 'attention_{}'.format(i), attention)

            input_shape = output_shape

        # Decoder output shape (before softmax)
        self.output_shape = input_shape

        self.fc_last = Linear(self.output_shape[2], self.task.TargetVocabSize)

    def forward(self, encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state=None):
        """

        Args:
            encoder_out (tuple):
                output: (batch_size, src_seq_len, encoder_out_channels) of float32
                output add source embedding: same shape as output
            src_lengths: (batch_size,) of long
            trg_tokens: (batch_size, trg_seq_len) of int32
            trg_lengths: (batch_size,) of long
            incremental_state: Incremental states for decoding. TODO

        Returns:

        """
        x = trg_tokens
        logging.debug('Decoder input shape: {}'.format(list(x.shape)))

        x = self._embed_tokens(x, incremental_state) + self.embed_positions(x, incremental_state)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        target_embedding = x

        logging.debug('Decoder input shape after embedding: {}'.format(list(x.shape)))
        avg_attn_scores = None
        num_attn_layers = len(self._attentions)
        for i, (layer, projection, attention) in enumerate(zip(self._net, self._projections, self._attentions)):
            residual = x if projection is None else projection(x)

            x = layer(x, trg_lengths)

            # Attention layer.
            # TODO: Add attention layer here
            #   x, attn_scores = attention(x, target_embedding, encoder_outs)
            # print('#', x.shape, target_embedding.shape, encoder_out[0].shape, encoder_out[1].shape, trg_lengths.shape)
            # x, attn_scores = attention(x, target_embedding, encoder_out, trg_lengths)
            # attn_scores = attn_scores / num_attn_layers
            # if avg_attn_scores is None:
            #     avg_attn_scores = attn_scores
            # else:
            #     avg_attn_scores.add_(attn_scores)

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
        return x, avg_attn_scores

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # Keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    @staticmethod
    def get_normalized_probs(net_output, log_probs=False):
        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def max_positions(self):
        return self.embed_positions.max_positions()


class ChildNet(nn.Module):
    def __init__(self, net_code, hparams):
        super().__init__()

        self.net_code = net_code
        self.hparams = hparams
        self.task = get_task(hparams.task)

        self.encoder = ChildEncoder(net_code[0], hparams)
        self.decoder = ChildDecoder(net_code[1], hparams, encoder_output_shape=self.encoder.output_shape)

    def forward(self, src_tokens, src_lengths, trg_tokens, trg_lengths):
        """

        Args:
            src_tokens: (batch_size, src_seq_len) of int32
            src_lengths: (batch_size,) of long
            trg_tokens: (batch_size, trg_seq_len) of int32
            trg_lengths: (batch_size,) of long

        Returns:
            (batch_size, seq_len, tgt_vocab_size) of float32
        """
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(encoder_out, src_lengths, trg_tokens, trg_lengths)

        return decoder_out

    def get_normalized_probs(self, net_output, log_probs=False):
        return self.decoder.get_normalized_probs(net_output, log_probs)

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def max_encoder_positions(self):
        return self.encoder.max_positions()

    def max_decoder_positions(self):
        return self.decoder.max_positions()

    def num_parameters(self):
        """Number of parameters."""
        return sum(p.data.numel() for p in self.parameters())
