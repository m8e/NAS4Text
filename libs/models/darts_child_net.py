#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from .child_net_base import ChildNetBase, EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase
from ..tasks import get_task
from ..layers.common import *
from ..layers.build_block import build_block
from ..layers.grad_multiply import GradMultiply

__author__ = 'fyabc'


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

        # TODO: Build alphas.
        self.alphas = None

        # TODO: Add DartsBlock here.

    def reorder_encoder_out(self, encoder_out, new_order):
        return encoder_out

    def forward(self, src_tokens, src_lengths=None):
        pass


class DartsChildDecoder(ChildIncrementalDecoderBase):
    def __init__(self, hparams, embed_tokens):
        # [NOTE]: Does not use net code, pass ``None``.
        super().__init__(None, hparams)

        # Decoder input shape (after embedding)
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.trg_embedding_size])

        self._build_embedding(embed_tokens)

        self.layers = nn.ModuleList()

        # TODO: Build alphas.
        self.alphas = None

        # TODO: Add DartsBlock here.

    def forward(self, encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state=None):
        pass


@ChildNetBase.register_child_net
class DartsChildNet(EncDecChildNet):
    def __init__(self, hparams):
        super().__init__(None, hparams)

        src_embed_tokens, trg_embed_tokens = self._build_embed_tokens()

        self.encoder = DartsChildEncoder(self.hparams, src_embed_tokens)
        self.decoder = DartsChildDecoder(self.hparams, trg_embed_tokens)


class DartsNet:
    pass
