#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

import torch as th
import torch.nn as nn

from .child_net_base import EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase
from ..layers.nao_layer import NAOLayer
from ..layers.nas_controller import NASController


class NAOChildEncoder(ChildEncoderBase):
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

        for i in range(hparams.num_encoder_layers):
            # [NOTE]: Shape not changed here.
            self.layers.append(NAOLayer(hparams, input_shape, in_encoder=True))

        self._init_post(input_shape)

    @property
    def num_layers(self):
        return len(self.layers)

    def reorder_encoder_out(self, encoder_out, new_order):
        raise RuntimeError('This method must not be called')

    def forward(self, *input):
        raise RuntimeError('This method must not be called')


class NAOChildDecoder(ChildIncrementalDecoderBase):
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

        for i in range(hparams.num_decoder_layers):
            # [NOTE]: Shape not changed here.
            self.layers.append(NAOLayer(hparams, input_shape, in_encoder=False))

        self._init_post(input_shape)

    @property
    def num_layers(self):
        return len(self.layers)

    def forward(self, *input):
        raise RuntimeError('This method must not be called')


class NAOChildNet(EncDecChildNet):
    """The class of NAO child net.

    [NOTE]: This class is just a "container" of shared weights, the forward and backward methods will not be called.
    """
    def __init__(self, hparams):
        super().__init__(None, hparams)

        src_embed_tokens, trg_embed_tokens = self._build_embed_tokens()

        self.encoder = NAOChildEncoder(hparams, src_embed_tokens)
        self.decoder = NAOChildDecoder(hparams, trg_embed_tokens)


class NAOController(NASController):
    def __init__(self, hparams):
        super().__init__(hparams)

        # The model which contains shared weights.
        self.shared_weights = NAOChildNet(hparams)

    # TODO: Apply shared weights into ppp of layer, node, op.
    # TODO: Apply shared weights into ops.

    def get_weight(self, in_encoder, layer_id, index, input_index, op_type, **kwargs):
        # [NOTE]: ENAS sharing style.
        pass

    def cuda(self, device=None):
        self.shared_weights.cuda(device)
        return self

    def generate_arch(self, n, num_nodes):
        pass
