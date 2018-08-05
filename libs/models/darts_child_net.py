#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .child_net_base import ChildNetBase, EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase
from ..layers.common import *
from ..layers.darts_layer import DartsLayer
from ..layers.build_block import build_block
from ..layers.grad_multiply import GradMultiply

__author__ = 'fyabc'


def _init_alphas(in_encoder):
    # TODO: Build alphas.
    pass


class DartsChildEncoder(ChildEncoderBase):
    def __init__(self, hparams, embed_tokens):
        # [NOTE]: Does not use net code, pass ``None``.
        super().__init__(None, hparams)

        # Encoder input shape (after embedding).
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.src_embedding_size])

        # Embeddings.
        self._build_embedding(embed_tokens)

        # The main encoder network.
        self.layers = nn.ModuleList()
        input_shape = self.input_shape

        # [NOTE]: Alphas are shared between all layers.
        self.alphas = _init_alphas(in_encoder=True)

        for i in range(hparams.num_nodes):
            # [NOTE]: Shape not changed here.
            self.layers.append(DartsLayer(hparams, input_shape, in_encoder=True))

        self._init_post(input_shape)

    @property
    def num_layers(self):
        return len(self.layers)

    def reorder_encoder_out(self, encoder_out, new_order):
        return encoder_out

    def forward(self, src_tokens, src_lengths=None):
        x, src_mask, source_embedding = self._fwd_pre(src_tokens, src_lengths)

        input_list = [x, x]
        for i in range(self.num_layers):
            layer = self.layers[i]
            output = layer(
                input_list[-1], input_list[-2],
                weights=F.softmax(self.alphas, dim=-1),

                lengths=src_lengths, mask=src_mask)
            input_list.append(output)

            logging.debug('Encoder layer {} output shape: {}'.format(i, list(output.shape)))
        x = input_list[-1]

        return self._fwd_post(x, src_mask, source_embedding)


class DartsChildDecoder(ChildIncrementalDecoderBase):
    def __init__(self, hparams, embed_tokens):
        # [NOTE]: Does not use net code, pass ``None``.
        super().__init__(None, hparams)

        # Decoder input shape (after embedding)
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.trg_embedding_size])

        self._build_embedding(embed_tokens)

        # The main encoder network.
        self.layers = nn.ModuleList()
        input_shape = self.input_shape

        # [NOTE]: Alphas are shared between all layers.
        self.alphas = _init_alphas(in_encoder=False)

        for i in range(hparams.num_nodes):
            # [NOTE]: Shape not changed here.
            self.layers.append(DartsLayer(hparams, input_shape, in_encoder=False))

        self._init_post(input_shape)

    def forward(self, encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state=None):
        x, encoder_out, trg_mask, target_embedding, encoder_state_mean = self._fwd_pre(
            encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state
        )

        input_list = [x, x]
        for i in range(self.num_layers):
            layer = self.layers[i]

            output = layer(
                input_list[-1], input_list[-2],
                weights=F.softmax(self.alphas, dim=-1),

                lengths=trg_lengths, encoder_state=encoder_out, src_lengths=src_lengths,
                target_embedding=target_embedding, encoder_state_mean=encoder_state_mean,
                mask=trg_mask,
            )
            input_list.append(output)

            logging.debug('Decoder layer {} output shape: {}'.format(i, list(x.shape)))
        x = input_list[-1]

        return self._fwd_post(x, None)


@ChildNetBase.register_child_net
class DartsChildNet(EncDecChildNet):
    def __init__(self, hparams):
        super().__init__(None, hparams)

        src_embed_tokens, trg_embed_tokens = self._build_embed_tokens()

        self.encoder = DartsChildEncoder(self.hparams, src_embed_tokens)
        self.decoder = DartsChildDecoder(self.hparams, trg_embed_tokens)

    def update_weights(self):
        pass

    def update_alphas(self):
        pass

    def dump_net_code(self, branch=2):
        pass


class DartsNet:
    pass
