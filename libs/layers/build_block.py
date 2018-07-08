#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

from .base import ChildLayer

__author__ = 'fyabc'


class Node(nn.Module):
    """Block node.

    Input[1], Input[2], Op[1], Op[2], CombineOp => Output
    """

    def forward(self):
        pass


class BlockLayer(ChildLayer):
    """Block layer. Contains several nodes."""

    def __init__(self, hparams, in_encoder=True):
        super().__init__(hparams)
        self.in_encoder = in_encoder
        self.nodes = nn.ModuleList()

    def build(self, layer_code, input_shape):
        return input_shape

    def forward(self, input_, prev_input, lengths=None, encoder_state=None):
        pass


def build_block(layer_code, input_shape, hparams, in_encoder=True):
    block = BlockLayer(hparams, in_encoder)
    output_shape = block.build(layer_code, input_shape)

    return block, output_shape


__all__ = [
    'BlockLayer',
    'build_block',
]
