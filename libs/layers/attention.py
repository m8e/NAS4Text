#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Attention layer."""

import torch as th

from .multi_head_attention import SelfAttention
from .ppp import push_prepostprocessors
from ..utils.search_space import AttentionSpaces

__author__ = 'fyabc'


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

    space = AttentionSpaces[hparams.attn_space]

    batch_size, seq_length, input_size = input_shape
    num_heads = space.NumHeads[layer_code[1]]

    layer = SelfAttention(
        hparams=hparams,
        h=num_heads,
        d_model=input_size,
        d_ff=hparams.attn_d_ff,
        in_encoder=in_encoder,
        linear_bias=hparams.attn_linear_bias,
    )

    output_shape = th.Size([batch_size, seq_length, input_size])
    push_prepostprocessors(layer, layer_code[-2], layer_code[-1], input_shape, output_shape)

    return layer, output_shape
