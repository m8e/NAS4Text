#! /usr/bin/python
# -*- coding: utf-8 -*-

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .common import Linear

__author__ = 'fyabc'


def attention(query, key, value, mask=None, dropout=None):
    r"""Compute scaled dot-product attention.

    :math:`Attention(Q, K, V) = softmax( Q * K^T / \sqrt{d_k} ) * V`

    Args:
        query (batch_size, length_q, d_k):
        key (batch_size, length_kv, d_k):
        value (batch_size, length_kv, d_v):
        mask (batch_size, length_q, length_kv): TODO: confirm this shape.
        dropout:

    Returns:
        tuple
            Attention value (batch_size, length_q, d_v):
            p_attn (batch_size, length_q, length_kv): Attention probability distribution
    """

    d_k = query.size(-1)
    scores = th.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill_(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return th.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    r"""The module of multi-head attention.

    :math:`MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O`

    where :math:`head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)`

    where :math:`W_i^Q, W_i^K \in R^{d_{model} \times d_k}, W_i^V \in R^{d_{model} \times d_v}`

    and :math:`W^O \in R^{h * d_v \times d_{model}}`.

    Args:
        h: Number of heads.
        d_model: Model output size.
        dropout:

    Inputs: query, key, value, mask
        - **query** (batch_size, length_q, d_model):
        - **key** (batch_size, length_kv, d_model):
        - **value** (batch_size, length_kv, d_model):
        - **mask** (?) or None:

    Output:
        - **output** (batch_size, length_q, d_model):
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        assert d_model % h == 0

        # [NOTE]: We assume that d_v always == d_k.
        self.d_k = d_model // h
        self.h = h

        # 4 Weights matrices.
        self.linears = nn.ModuleList([Linear(d_model, d_model) for _ in range(4)])

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        num_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class SelfAttention(MultiHeadAttention):
    """Wraps multi-head attention into self attention."""
    def forward(self, x, mask=None):
        return super().forward(x, x, x, mask=mask)


__all__ = [
    'MultiHeadAttention',
    'SelfAttention',
]
