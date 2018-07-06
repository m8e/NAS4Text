#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from .utils.data_processing import LanguagePairDataset
from .tasks import get_task
from .layers.common import *
from .build_block import build_block

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
        self.num_layers = 0
        input_shape = self.input_shape
        # self.fc1 = Linear(input_shape[2], 128)
        # input_shape = th.Size([1, 1, 128])
        for i, layer_code in enumerate(code):
            layer, output_shape = build_block(layer_code, input_shape, self.hparams, in_encoder=True)
            setattr(self, 'layer_{}'.format(i), layer)

            input_shape = output_shape
            self.num_layers += 1

        # TODO: Build blocks.

    def forward(self):
        # TODO
        pass

    def upgrade_state_dict(self, state_dict):
        return state_dict

    def max_positions(self):
        return self.embed_positions.max_positions()

    def get_layer(self, i):
        return getattr(self, 'layer_{}'.format(i))

    def get_layers(self):
        return [self.get_layer(i) for i in range(self.num_layers)]
