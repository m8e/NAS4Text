#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Attention layer.

Layer code:
[Attention, NumHeads, ..., Preprocessors, Postprocessors]

# TODO: GlobalAttention?, WindowSize, ...
"""

import torch as th

from .multi_head_attention import SelfAttention
from .ppp import PPPSpace, push_prepostprocessors

__author__ = 'fyabc'


class AttentionSpaceBase:
    # TODO: How to ensure the assertion "input_hidden_size % num_heads == 0" to be always True?
    NumHeads = [2, 4, 8, 16]
    Preprocessors = PPPSpace.Preprocessors
    Postprocessors = PPPSpace.Postprocessors


class AttentionSpaceLarge(AttentionSpaceBase):
    pass


Spaces = {
    'base': AttentionSpaceBase,
    'large': AttentionSpaceLarge,
}


def build_attention(layer_code, input_shape, hparams, in_encoder=True):
    """

    Args:
        layer_code:
        input_shape: torch.Size object
            Shape of input tensor, expect (batch_size, seq_len, input_size)
        hparams:
        in_encoder: bool
            Indicates if the layer is in encoder or decoder

    Returns: layer, output_shape
        output_shape: torch.Size object
            Shape of output tensor, (batch_size, seq_len, hidden_size * num_directions)
    """

    if len(layer_code) == 2:
        # Old-style layer code (without pre/post processing)
        layer_code += [0, 0]
    else:
        assert len(layer_code) == 4, 'Layer code must have length of 2 or 4, got {}'.format(len(layer_code))

    space = Spaces[hparams.attn_space]

    batch_size, seq_length, input_size = input_shape
    num_heads = space.NumHeads[layer_code[1]]

    layer = SelfAttention(
        hparams=hparams,
        h=num_heads,
        d_model=input_size,
        dropout=hparams.dropout,
        in_encoder=in_encoder,
    )

    output_shape = th.Size([batch_size, seq_length, input_size])
    push_prepostprocessors(layer, layer_code[-2], layer_code[-1], input_shape, output_shape)

    return layer, output_shape
