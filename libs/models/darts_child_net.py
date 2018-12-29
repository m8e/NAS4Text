#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .child_net_base import ChildNetBase, EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase, \
    forward_call, ParalleledChildNet
from ..layers.darts_layer import DartsLayer
from ..utils import common

__author__ = 'fyabc'


def _init_alphas(darts_layer: DartsLayer):
    """

    [NOTE]: Alphas are Variables, so they will NOT be registered into the module as parameters.

    :param darts_layer:
    :return:
    """
    I = darts_layer.num_input_nodes
    N = darts_layer.num_nodes
    num_edges = I * N + N * (N - 1) // 2
    num_ops = len(darts_layer.supported_ops())

    # # FIXME: Return parameter or variable here?
    # return nn.Parameter(1e-3 * th.randn(num_edges, num_ops))
    return common.make_variable(1e-3 * th.randn(num_edges, num_ops), volatile=False, cuda=True, requires_grad=True)


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

        for i in range(hparams.num_encoder_layers):
            # [NOTE]: Shape not changed here.
            self.layers.append(DartsLayer(hparams, input_shape, in_encoder=True))

        # [NOTE]: Alphas are shared between all layers.
        self.alphas = _init_alphas(self.layers[0])

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

    def dump_net_code(self, branch=2):
        return self.layers[0].dump_net_code(F.softmax(self.alphas, dim=-1).data.cpu().numpy(), branch)


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

        for i in range(hparams.num_decoder_layers):
            # [NOTE]: Shape not changed here.
            self.layers.append(DartsLayer(hparams, input_shape, in_encoder=False))

        self.alphas = _init_alphas(self.layers[0])

        self._init_post(input_shape)

    @property
    def num_layers(self):
        return len(self.layers)

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
                target_embedding=target_embedding if self.hparams.connect_trg_emb else None,
                encoder_state_mean=encoder_state_mean, mask=trg_mask, src_mask=encoder_out['src_mask'],
            )
            input_list.append(output)

            logging.debug('Decoder layer {} output shape: {}'.format(i, list(x.shape)))
        x = input_list[-1]

        return self._fwd_post(x, None)

    def _contains_lstm(self):
        return any(o[0] == 'LSTM' for o in self.layers[0].supported_ops())

    def dump_net_code(self, branch=2):
        return self.layers[0].dump_net_code(F.softmax(self.alphas, dim=-1).data.cpu().numpy(), branch)


@ChildNetBase.register_child_net
class DartsChildNet(EncDecChildNet):
    def __init__(self, hparams):
        super().__init__(None, hparams)

        src_embed_tokens, trg_embed_tokens = self._build_embed_tokens()

        self.encoder = DartsChildEncoder(self.hparams, src_embed_tokens)
        self.decoder = DartsChildDecoder(self.hparams, trg_embed_tokens)

    def arch_parameters(self):
        return [self.encoder.alphas, self.decoder.alphas]

    def dump_net_code(self, branch=2):
        return {
            'Type': 'BlockChildNet',
            'Global': {},
            'Blocks': {
                'enc1': self.encoder.dump_net_code(branch),
                'dec1': self.decoder.dump_net_code(branch),
            },
            'Layers': [
                ['enc1' for _ in range(self.encoder.num_layers)],
                ['dec1' for _ in range(self.decoder.num_layers)],
            ],
        }


class ParalleledDartsChildNet(ParalleledChildNet):
    arch_parameters = forward_call('arch_parameters')
    dump_net_code = forward_call('dump_net_code')
