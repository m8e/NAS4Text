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
from .utils import common
from .layers.common import Linear, Embedding, PositionalEmbedding
from .layers.net_code import NetCodeEnum
from .layers.lstm import build_lstm
from .layers.cnn import build_cnn
from .layers.attention import build_attention
from .layers.multi_head_attention import EncDecAttention
from .layers.grad_multiply import GradMultiply

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

        # Value from decoder
        self.num_attention_layers = None

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

            projection = Linear(input_shape[2], output_shape[2]) if input_shape != output_shape else None
            setattr(self, 'projection_{}'.format(i), projection)
            input_shape = output_shape
            self.num_layers += 1

        # Encoder output shape
        self.output_shape = input_shape

        self.fc2 = Linear(self.output_shape[2], hparams.src_embedding_size)

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
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_{}'.format(i))
            projection = getattr(self, 'projection_{}'.format(i))

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
        x = self.fc2(x)

        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + source_embedding) * math.sqrt(0.5)

        logging.debug('Encoder output shape: {} & {}'.format(list(x.shape), list(y.shape)))
        return x, y

    def upgrade_state_dict(self, state_dict):
        return state_dict

    def max_positions(self):
        return self.embed_positions.max_positions()


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
        self.num_layers = 0

        input_shape = self.input_shape
        for i, layer_code in enumerate(code):
            layer, output_shape = _code2layer(layer_code, input_shape, self.hparams, in_encoder=False)
            setattr(self, 'layer_{}'.format(i), layer)

            projection = Linear(input_shape[2], output_shape[2]) if input_shape != output_shape else None
            setattr(self, 'projection_{}'.format(i), projection)

            attention = EncDecAttention(8, output_shape[2], hparams.trg_embedding_size, hparams.src_embedding_size,
                                        dropout=hparams.dropout, in_encoder=False)
            setattr(self, 'attention_{}'.format(i), attention)

            input_shape = output_shape
            self.num_layers += 1

        # Decoder output shape (before softmax)
        self.output_shape = input_shape

        self.fc2 = Linear(self.output_shape[2], hparams.decoder_out_embedding_size)

        if hparams.share_input_output_embedding:
            assert hparams.trg_embedding_size == hparams.decoder_out_embedding_size, \
                'Shared embed weights implies same dimensions out_embedding_size={} vs trg_embedding_size={}'.format(
                    hparams.decoder_out_embedding_size, hparams.trg_embedding_size)
            self.fc_last = nn.Linear(hparams.decoder_out_embedding_size, self.task.TargetVocabSize)
            self.fc_last.weight = self.embed_tokens.weight
        else:
            self.fc_last = Linear(hparams.decoder_out_embedding_size, self.task.TargetVocabSize)

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
            Attention scores: (batch_size, num_heads(8), trg_seq_len, src_seq_len) of float32
        """

        # split and (transpose) encoder outputs
        encoder_out = self._split_encoder_out(encoder_out, incremental_state)

        x = trg_tokens
        logging.debug('Decoder input shape: {}'.format(list(x.shape)))

        x = self._embed_tokens(x, incremental_state) + self.embed_positions(x, incremental_state)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        target_embedding = x

        logging.debug('Decoder input shape after embedding: {}'.format(list(x.shape)))
        avg_attn_scores = None
        num_attn_layers = self.num_layers   # TODO: Explain why include layers without attention (None)?
        for i in range(self.num_layers):
            layer = getattr(self, 'layer_{}'.format(i))
            projection = getattr(self, 'projection_{}'.format(i))
            attention = getattr(self, 'attention_{}'.format(i))

            residual = x if projection is None else projection(x)

            x = layer(x, trg_lengths)

            # Attention layer.
            x, attn_scores = attention(x, target_embedding, encoder_out, trg_lengths)
            attn_scores = attn_scores / num_attn_layers
            if avg_attn_scores is None:
                avg_attn_scores = attn_scores
            else:
                avg_attn_scores.add_(attn_scores)

            # Residual connection.
            # If sequence length changed, add 1x1 convolution ([NOTE]: The layer must provide it).
            # We cannot determine sequence length when building the module, so test them here.
            if x.shape[1] != residual.shape[1]:
                residual = layer.residual_conv(residual.transpose(1, 2)).transpose(1, 2)
            x = (x + residual) * math.sqrt(0.5)

            logging.debug('Decoder layer {} output shape: {}'.format(i, list(x.shape)))

        # Project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        x = self.fc_last(x)

        logging.debug('Decoder output shape: {} & {}'.format(list(x.shape), list(avg_attn_scores.shape)))
        return x, avg_attn_scores

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # Keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _split_encoder_out(self, encoder_out, incremental_state):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = common.get_incremental_state(self, incremental_state, 'encoder_out')
        if cached_result is not None:
            return cached_result

        # # transpose only once to speed up attention layers
        # # TODO: NOT do transpose here because of multi-head attention
        # encoder_a, encoder_b = encoder_out
        # encoder_a = encoder_a.transpose(1, 2).contiguous()
        # result = (encoder_a, encoder_b)
        result = encoder_out

        if incremental_state is not None:
            common.set_incremental_state(self, incremental_state, 'encoder_out', result)
        return result

    @staticmethod
    def get_normalized_probs(net_output, log_probs=False):
        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def upgrade_state_dict(self, state_dict):
        return state_dict

    def max_positions(self):
        return self.embed_positions.max_positions()

    @property
    def num_attention_layers(self):
        return sum(getattr(self, 'attention_{}'.format(i)) is not None for i in range(self.num_layers))


class ChildNet(nn.Module):
    def __init__(self, net_code, hparams):
        super().__init__()

        self.net_code = net_code
        self.hparams = hparams
        self.task = get_task(hparams.task)

        self.encoder = ChildEncoder(net_code[0], hparams)
        self.decoder = ChildDecoder(net_code[1], hparams, encoder_output_shape=self.encoder.output_shape)
        self.encoder.num_attention_layers = self.decoder.num_attention_layers

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

    def encode(self, src_tokens, src_lengths):
        return self.encoder(src_tokens, src_lengths)

    def decode(self, encoder_out, src_lengths, trg_tokens, trg_lengths):
        return self.decoder(encoder_out, src_lengths, trg_tokens, trg_lengths)

    def get_normalized_probs(self, net_output, log_probs=False):
        return self.decoder.get_normalized_probs(net_output, log_probs)

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def max_encoder_positions(self):
        return self.encoder.max_positions()

    def max_decoder_positions(self):
        return self.decoder.max_positions()

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from state_dict into this module and
        its descendants.

        Overrides the method in nn.Module; compared with that method this
        additionally "upgrades" state_dicts from old checkpoints.
        """
        state_dict = self.upgrade_state_dict(state_dict)
        super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        state_dict = self.encoder.upgrade_state_dict(state_dict)
        state_dict = self.decoder.upgrade_state_dict(state_dict)
        return state_dict

    def num_parameters(self):
        """Number of parameters."""
        return sum(p.data.numel() for p in self.parameters())


def _forward_call(method_name):
    def _method(self, *args, **kwargs):
        return getattr(self.module, method_name)(*args, **kwargs)
    return _method


class ParalleledChildNet(nn.DataParallel):
    encode = _forward_call('encode')
    decode = _forward_call('decode')
    get_normalized_probs = _forward_call('get_normalized_probs')
    get_targets = _forward_call('get_targets')
    max_encoder_positions = _forward_call('max_encoder_positions')
    max_decoder_positions = _forward_call('max_decoder_positions')
    upgrade_state_dict = _forward_call('upgrade_state_dict')
    num_parameters = _forward_call('num_parameters')
