#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Multi-head attention layer."""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .base import ChildLayer
from .common import Linear
from ..utils import common
from ..utils.data_processing import LanguagePairDataset

__author__ = 'fyabc'


def attention(query, key, value, mask=None, dropout=None):
    r"""Compute scaled dot-product attention.

    :math:`Attention(Q, K, V) = softmax( Q * K^T / \sqrt{d_k} ) * V`

    Args:
        query (batch_size, num_heads, length_q, d_k):
        key (batch_size, num_heads, length_kv, d_k):
        value (batch_size, num_heads, length_kv, d_v):
        mask (batch_size, 1, 1 or length_q, length_kv):
        dropout:

    Returns:
        tuple
            Attention value (batch_size, num_heads, length_q, d_v):
            p_attn (batch_size, num_heads, length_q, length_kv): Attention probability distribution
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
        window: Local attention window size, None means global attention.

    Inputs: query, key, value, mask
        - **query** (batch_size, length_q, d_model):
        - **key** (batch_size, length_kv, d_model):
        - **value** (batch_size, length_kv, d_model):
        - **mask** (batch_size, 1, 1 or length_q, length_kv) or None:

    Output:
        - **output** (batch_size, length_q, d_model):
    """

    def __init__(self, h, d_model, **kwargs):
        super().__init__()

        hparams = kwargs.pop('hparams', None)

        assert d_model % h == 0

        # [NOTE]: We assume that d_v always == d_k.
        self.d_k = d_model // h
        self.h = h
        self.window = window = kwargs.pop('window', None)

        assert window is None or (isinstance(window, int) and window % 2 == 1), \
            'Local attention window size must be None or an odd number'

        # 4 Weights matrices.
        linear_bias = kwargs.pop('linear_bias', True)
        self.linears = nn.ModuleList([
            Linear(d_model, d_model, hparams=hparams, bias=linear_bias)
            for _ in range(4)])

        self.attn = None
        self.dropout = nn.Dropout(p=kwargs.pop('dropout', 0.1))

    def forward(self, query, key, value, mask=None):
        num_batches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))]

        # print('$d_k: {}, h: {}'.format(self.d_k, self.h))
        # print('#Q: {}, K: {}, V: {}, M: {}'.format(query.shape, key.shape, value.shape, mask.shape))

        key_local = None

        # TODO: Implement local attention.
        if self.window is not None:
            pass

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.h * self.d_k)

        result = self.linears[-1](x)

        return result


def _mask_from_lengths(x, lengths, layer, subsequent_mask=False, maxlen=None):
    if lengths is None:
        return None

    if maxlen is None:
        maxlen = x.size(1)

    left_pad = LanguagePairDataset.LEFT_PAD_SOURCE if layer.in_encoder else LanguagePairDataset.LEFT_PAD_TARGET
    mask = common.mask_from_lengths(lengths, left_pad=left_pad, max_length=maxlen, cuda=True)

    # Same mask applied to whole query sequence.
    mask = mask.unsqueeze(1)

    # Apply subsequent mask.
    if subsequent_mask:
        mask = mask & common.make_variable(
            common.subsequent_mask(x.size(1)),
            cuda=True,
        )

    # Same mask applied to all h heads.
    mask = mask.unsqueeze(1)

    return mask


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, **kwargs):
        super().__init__()

        hparams = kwargs.pop('hparams', None)
        linear_bias = kwargs.pop('linear_bias', True)

        self.w_1 = Linear(d_model, d_ff, hparams=hparams, bias=linear_bias)
        self.w_2 = Linear(d_ff, d_model, hparams=hparams, bias=linear_bias)
        self.dropout = nn.Dropout(kwargs.pop('dropout', 0.1))

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SelfAttention(ChildLayer):
    """Wraps multi-head attention into self attention.

    Inputs: x, lengths
        - **x** (batch_size, length, d_model):
        - **lengths** (batch_size,) or None:

    Output:
        - **output** (batch_size, length, d_model):
    """

    def __init__(self, hparams, h, d_model, d_ff, **kwargs):
        super().__init__(hparams)

        linear_bias = kwargs.pop('linear_bias', True)

        self.in_encoder = kwargs.pop('in_encoder', True)
        self.attention = MultiHeadAttention(
            h, d_model, dropout=kwargs.pop('dropout', 0.1), hparams=hparams, linear_bias=linear_bias)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout=kwargs.pop('ffn_dropout', 0.1), hparams=hparams, linear_bias=linear_bias)

        # [NOTE]: The encoder-decoder attention layer may be inside this attention layer.
        # Used in decoder.
        self.encdec_attention = None
        self.attn_scores = None

    def add_encdec_attention(self, layer, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        self.encdec_attention = lambda x: layer(x, *args, **kwargs)

    def forward(self, x, lengths=None, **kwargs):
        """

        Args:
            x:
            lengths:
            **kwargs:

        Returns:

        Notes:
            The self-attention layer contains a multi-head attention layer and a position-wise feed-forward layer.
            Each need to be preprocessed and postprocessed.
        """

        attn_input = self.preprocess(x)
        mask = _mask_from_lengths(attn_input, lengths, self, subsequent_mask=True)
        attn_result = self.attention(attn_input, attn_input, attn_input, mask=mask)
        attn_result = self.postprocess(attn_result, x)

        if self.encdec_attention is not None:
            encdec_input = self.preprocess(attn_result)
            encdec_result, self.attn_scores = self.encdec_attention(encdec_input)
            attn_result = self.postprocess(encdec_result, attn_result)

        ff_input = self.preprocess(attn_result)
        result = self.feed_forward(ff_input)
        result = self.postprocess(result, attn_result)

        return result


class EncDecAttention(nn.Module):
    """Encoder-decoder attention module, modified for different input sizes."""
    def __init__(self, h, conv_channels, trg_emb_size, src_emb_size,
                 **kwargs):
        super().__init__()

        hparams = kwargs.pop('hparams', None)
        linear_bias = kwargs.pop('linear_bias', True)

        assert trg_emb_size % h == 0

        self.h = h
        d_model = trg_emb_size

        # [NOTE]: We assume that d_v always == d_k.
        self.d_k = d_model // h
        self.h = h

        # 4 Weights matrices.
        self.linears = nn.ModuleList([
            Linear(conv_channels, d_model, hparams=hparams, bias=linear_bias),
            Linear(src_emb_size, d_model, hparams=hparams, bias=linear_bias),
            Linear(src_emb_size, d_model, hparams=hparams, bias=linear_bias),
            Linear(d_model, conv_channels, hparams=hparams, bias=linear_bias),
        ])

        self.attn = None
        self.dropout = nn.Dropout(p=kwargs.pop('dropout', 0.1))
        self.in_encoder = kwargs.pop('in_encoder', True)
        self.hparams = hparams

    def forward(self, x, target_embedding, encoder_outs, src_lengths=None):
        """

        Args:
            x: (batch_size, trg_seq_len, conv_channels) of float32
            target_embedding: (batch_size, trg_seq_len, trg_emb_size) of float32
            encoder_outs (tuple):
                output: (batch_size, src_seq_len, src_emb_size) of float32
                output add source embedding: same shape as output
            src_lengths: (batch_size,) of long

        Returns:
            output: (batch_size, trg_seq_len, conv_channels) of float32
            attn_score: (batch_size, trg_seq_len, src_seq_len) of float32
        """
        # Mask: (batch_size, 1, src_seq_len)
        mask = _mask_from_lengths(x, src_lengths, self, subsequent_mask=False, maxlen=encoder_outs[0].size(1))
        num_batches = x.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # conv_channel -> trg_emb_size (+ target_embedding)
        x = self.linears[0](x)
        if target_embedding is not None:
            x = (x + target_embedding) * math.sqrt(0.5)
        query = x.view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linears[1](encoder_outs[0]).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](encoder_outs[1]).view(num_batches, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        self.attn = self.attn.mean(dim=1)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.h * self.d_k)

        x = self.linears[-1](x)

        return x, self.attn


__all__ = [
    'MultiHeadAttention',
    'SelfAttention',
    'EncDecAttention',
]
