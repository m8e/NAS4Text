#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
import torch.nn as nn

from .child_net_base import EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase
from ..layers.nao_layer import NAOLayer
from ..layers.nas_controller import NASController
from ..layers.net_code import NetCode
from ..layers.common import *


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


class NAO_EPD(nn.Module):
    # TODO: Change them into hparams? Or compute them from search space?
    encoder_vocab_size = 10
    decoder_vocab_size = 10
    encoder_emb_size = 32

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.encoder_emb = Embedding(self.encoder_vocab_size, self.encoder_emb_size, padding_idx=None, hparams=hparams)
        # self.encoder = nn.LSTM(
        #     # TODO
        # )
        self.predictor = None
        self.decoder = None

    def forward(self, encoder_input, encoder_target, decoder_target):
        pass


class NAOController(NASController):
    def __init__(self, hparams):
        super().__init__(hparams)

        # The model which contains shared weights.
        self.shared_weights = NAOChildNet(hparams)
        self._supported_ops_cache = {
            True: self._reversed_supported_ops(self.shared_weights.encoder.layers[0].supported_ops()),
            False: self._reversed_supported_ops(self.shared_weights.decoder.layers[0].supported_ops()),
        }

        # EPD.
        self.epd = NAO_EPD(hparams)

    @staticmethod
    def _reversed_supported_ops(supported_ops):
        return {
            (op_name, tuple(op_args)): i
            for i, (op_name, op_type, op_args) in enumerate(supported_ops)
        }

    def _codec(self, in_encoder):
        return self.shared_weights.encoder if in_encoder else self.shared_weights.decoder

    def _layer(self, in_encoder, layer_id):
        return self._codec(in_encoder).layers[layer_id]

    def get_weight(self, in_encoder, layer_id, index, input_index, op_code, **kwargs):
        # [NOTE]: ENAS sharing style, same as DARTS sharing style.
        op_args = kwargs.pop('op_args', [])
        op_idx = self._supported_ops_cache[in_encoder].get((op_code, tuple(op_args)), None)
        if op_idx is None:
            raise RuntimeError('The op type {} and op args {} does not exist in the controller'.format(
                op_code, op_args))
        layer = self._layer(in_encoder, layer_id)

        return layer.edges[layer.offsets[(input_index, index)]].ops[op_idx]

    def get_node_ppp(self, in_encoder, layer_id, index, **kwargs):
        layer = self._layer(in_encoder, layer_id)
        ppp = layer.node_ppp_list[index - layer.num_input_nodes]
        return {
            'pre': ppp.preprocessors,
            'post': ppp.postprocessors,
            'residual_projection': ppp.residual_projection,
        }

    def get_block_ppp(self, in_encoder, layer_id, **kwargs):
        layer = self._layer(in_encoder, layer_id)
        return {
            'pre': layer.preprocessors,
            'post': layer.postprocessors,
            'residual_projection': layer.residual_projection,
        }

    def cuda(self, device=None):
        self.shared_weights.cuda(device)
        self.epd.cuda(device)
        return self

    def _generate_block(self, layer: NAOLayer):
        result = []
        num_input_nodes = layer.num_input_nodes
        num_total_nodes = layer.num_total_nodes
        in_encoder = layer.in_encoder
        supported_ops = list(self._supported_ops_cache[in_encoder].keys())
        supported_ops_idx = list(range(len(supported_ops)))

        result.extend([[None for _ in range(2 * num_input_nodes + 1)] for _ in range(num_input_nodes)])

        for j in range(num_input_nodes, num_total_nodes):
            edges = [np.random.randint(0, j) for _ in range(2)]
            ops = []
            for _ in range(2):
                op_name, op_args = supported_ops[np.random.choice(supported_ops_idx)]
                ops.append([op_name] + list(op_args))

            result.append(
                edges +
                ops +
                [layer.node_combine_op] +
                layer.node_ppp_code
            )

        result.append({
            'preprocessors': layer.ppp_code[0],
            'postprocessors': layer.ppp_code[1],
        })

        return result

    def _template_net_code(self, e, d):
        return NetCode({
            'Type': 'BlockChildNet',
            'Global': {},
            'Blocks': {
                'enc1': e,
                'dec1': d,
            },
            'Layers': [
                ['enc1' for _ in range(self.shared_weights.encoder.num_layers)],
                ['dec1' for _ in range(self.shared_weights.decoder.num_layers)],
            ]
        })

    def generate_arch(self, n):
        enc0 = self.shared_weights.encoder.layers[0]
        dec0 = self.shared_weights.decoder.layers[0]

        return [self._template_net_code(self._generate_block(enc0), self._generate_block(dec0)) for _ in range(n)]

    def parse_arch_to_seq(self, arch, branch_length):
        # TODO
        pass

    def predict(self, topk_arches):
        pass
