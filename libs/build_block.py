#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn

__author__ = 'fyabc'


class Node(nn.Module):
    """Block node.

    Input[1], Input[2], Op[1], Op[2], CombineOp => Output
    """

    def forward(self):
        pass


class BlockLayer(nn.Module):
    """Block layer. Contains several nodes."""

    def forward(self):
        pass


def build_block(layer_code, input_shape, hparams, in_encoder=True):
    return None, None


__all__ = [
    'BlockLayer',
    'build_block',
]
