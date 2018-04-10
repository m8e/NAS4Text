#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Common used layers."""

import math

import torch.nn as nn

from .learned_positional_embedding import LearnedPositionalEmbedding

__author__ = 'fyabc'


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Weight-normalized Embedding layer"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
    m.weight.data.normal_(0, 0.1)
    return m


__all__ = [
    'Embedding',
    'Linear',
    'PositionalEmbedding',
]
