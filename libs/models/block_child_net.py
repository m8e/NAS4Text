#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

import torch as th
import torch.nn as nn

from .child_net_base import ChildNetBase, EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase
from ..layers.build_block import build_block

__author__ = 'fyabc'


class BlockChildEncoder(ChildEncoderBase):
    def __init__(self, code, hparams, embed_tokens, controller=None):
        super().__init__(code, hparams)

        # Encoder input shape (after embedding).
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.src_embedding_size])

        # Embeddings.
        self._build_embedding(embed_tokens)

        # The main encoder network.
        self.layers = nn.ModuleList()
        input_shape = self.input_shape
        # self.fc1 = Linear(input_shape[2], 128)
        # input_shape = th.Size([1, 1, 128])
        for i, layer_code in enumerate(code):
            layer, output_shape = build_block(layer_code, input_shape, self.hparams,
                                              in_encoder=True, controller=controller)
            self.layers.append(layer)

            input_shape = output_shape

        self._init_post(input_shape)

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
        x, src_mask, source_embedding = self._fwd_pre(src_tokens, src_lengths)

        input_list = [x, x]
        for i in range(self.num_layers):
            layer = self.layers[i]
            output = layer(input_list[-1], input_list[-2], lengths=src_lengths, mask=src_mask)
            input_list.append(output)

            logging.debug('Encoder layer {} output shape: {}'.format(i, list(output.shape)))
        x = input_list[-1]

        return self._fwd_post(x, src_mask, source_embedding)

    def reorder_encoder_out(self, encoder_out, new_order):
        # TODO: Implement this method.
        return encoder_out

    def upgrade_state_dict(self, state_dict):
        return state_dict


class BlockChildDecoder(ChildIncrementalDecoderBase):
    def __init__(self, code, hparams, embed_tokens, controller=None):
        super().__init__(code, hparams)

        # Decoder input shape (after embedding)
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.trg_embedding_size])

        self._build_embedding(embed_tokens)

        self.layers = nn.ModuleList()

        input_shape = self.input_shape
        for i, layer_code in enumerate(code):
            layer, output_shape = build_block(layer_code, input_shape, self.hparams,
                                              in_encoder=False, controller=controller)
            self.layers.append(layer)

            input_shape = output_shape

        self._init_post(input_shape)

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
            incremental_state: Incremental states for decoding.

        Returns:
            Output: (batch_size, trg_seq_len, trg_vocab_size) of float32
            Attention scores: (batch_size, trg_seq_len, src_seq_len) of float32
        """

        x, encoder_out, trg_mask, target_embedding, encoder_state_mean = self._fwd_pre(
            encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state
        )

        input_list = [x, x]
        for i in range(self.num_layers):
            layer = self.layers[i]

            output = layer(
                input_list[-1], input_list[-2],
                lengths=trg_lengths, encoder_state=encoder_out, src_lengths=src_lengths,
                target_embedding=target_embedding, encoder_state_mean=encoder_state_mean,
                mask=trg_mask,
            )
            input_list.append(output)

            logging.debug('Decoder layer {} output shape: {}'.format(i, list(x.shape)))
        x = input_list[-1]

        return self._fwd_post(x, None)

    def _contains_lstm(self):
        return any(l.contains_lstm() for l in self.layers)


@ChildNetBase.register_child_net
class BlockChildNet(EncDecChildNet):
    def __init__(self, net_code, hparams):
        super().__init__(net_code, hparams)

        self.controller = None
        self._build_nas_controller()

        src_embed_tokens, trg_embed_tokens = self._build_embed_tokens()

        self.encoder = BlockChildEncoder(net_code[0], hparams, src_embed_tokens, controller=self.controller)
        self.decoder = BlockChildDecoder(net_code[1], hparams, trg_embed_tokens, controller=self.controller)

    def _build_nas_controller(self):
        nas_algo = self.hparams.nas_algo
        if nas_algo is None:
            return
        elif nas_algo == 'darts':
            # DARTS algorithm does not use controller now.
            return
        else:
            raise NotImplementedError('This NAS algorithm {!r} is not implemented now'.format(nas_algo))
