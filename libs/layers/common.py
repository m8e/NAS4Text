#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Common used layers."""

import copy
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .learned_positional_embedding import LearnedPositionalEmbedding

__author__ = 'fyabc'


def clones(module, n):
    """Produce n identical layers."""
    # [NOTE]: This only support leaf modules created by user.
    # For example, the module returned from ``Linear`` function below cannot use this to clone.
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _compute_fans(shape):
    if len(shape) < 1:  # Just to avoid errors for constants.
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        raise NotImplementedError('Shape >= 3 not implemented')
    return fan_in, fan_out


def uniform_unit_scaling_initializer(tensor, scale=1.0, mode='fan_avg'):
    fan_in, fan_out = _compute_fans(tensor.shape)
    if mode == 'fan_in':
        scale /= max(1., fan_in)
    elif mode == 'fan_out':
        scale /= max(1., fan_out)
    else:   # mode == 'fan_avg'
        scale /= max(1., (fan_in + fan_out) / 2)
    limit = math.sqrt(3.0 * scale)
    nn.init.uniform(tensor, -limit, limit)


def Linear(in_features, out_features, bias=True, dropout=0, hparams=None):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)

    if hparams.initializer == 'original':
        m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
        if bias:
            m.bias.data.zero_()
        return nn.utils.weight_norm(m)
    elif hparams.initializer == 'uniform_unit_scaling':
        uniform_unit_scaling_initializer(m.weight, scale=hparams.initializer_gain)
        if bias:
            m.bias.data.zero_()
        return nn.utils.weight_norm(m)
    elif hparams.initializer == 'kaitao':
        m.weight.data.uniform_(-0.1, 0.1)
        if bias:
            m.bias.data.uniform_(-0.1, 0.1)
        return m
    elif hparams.initializer == 'kaitao_wn':
        m.weight.data.uniform_(-0.1, 0.1)
        if bias:
            m.bias.data.uniform_(-0.1, 0.1)
        return nn.utils.weight_norm(m)
    else:
        raise ValueError('Unknown initializer {!r}'.format(hparams.initializer))


def Embedding(num_embeddings, embedding_dim, padding_idx, hparams=None):
    """Weight-normalized Embedding layer"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    if hparams.initializer in ('original', 'kaitao', 'kaitao_wn'):
        m.weight.data.normal_(0, 0.1)
    elif hparams.initializer == 'uniform_unit_scaling':
        uniform_unit_scaling_initializer(m.weight, scale=hparams.initializer_gain)
    else:
        raise ValueError('Unknown initializer {!r}'.format(hparams.initializer))
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, hparams=None):
    m = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad)
    if hparams.initializer in ('original', 'kaitao', 'kaitao_wn'):
        m.weight.data.normal_(0, 0.1)
    elif hparams.initializer == 'uniform_unit_scaling':
        uniform_unit_scaling_initializer(m.weight, scale=hparams.initializer_gain)
    else:
        raise ValueError('Unknown initializer {!r}'.format(hparams.initializer))
    return m


class Identity(nn.Module):
    def forward(self, x):
        return x


class FairseqAttention(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None, hparams=None):
        super().__init__()

        self.hparams = hparams

        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim, hparams=hparams)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels, hparams=hparams)

        self.bmm = bmm if bmm is not None else th.bmm

    def forward(self, x, target_embedding, encoder_out, src_lengths=None):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output
        s = encoder_out[1].size(1)
        x = x * (s * math.sqrt(1.0 / s))

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        # TODO


def residual(x, input_, res_type='default'):
    if res_type == 'default':
        return x + input_
    elif res_type == 'original':
        return (x + input_) * math.sqrt(0.5)
    return x + input_


class NLCBatchNorm1d(nn.BatchNorm1d):
    """Batch normalization, applied on (N, L, C) input."""
    # TODO: Discuss on it (BN on RNN?), need test
    def forward(self, x, input_=None):
        if x.data.ndimension() == 3:
            return super().forward(x.transpose(1, 2)).transpose(1, 2)
        return super().forward(x)


class LayerNorm(nn.Module):
    """A Simple implementation of layer normalization, applied on (N, L, C) input."""

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(th.ones(num_features))
        self.beta = nn.Parameter(th.zeros(num_features))

    def forward(self, x, input_=None):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def __repr__(self):
        return '{name}({num_features}, eps={eps})'.format(name=self.__class__.__name__, **self.__dict__)


class MyDropout(nn.Dropout):
    """Same as dropout, used for postprocessors, that pass an extra argument to ``forward``."""
    def forward(self, x, input_=None):
        return super().forward(x)


class Residual(nn.Module):
    def __init__(self, res_type='default'):
        super().__init__()

        assert res_type in ('default', 'original'), 'Illegal residual type {!r}'.format(res_type)

        self.res_type = res_type

    def forward(self, x, input_):
        return residual(x, input_, self.res_type)


__all__ = [
    'clones',
    'Embedding',
    'Linear',
    'PositionalEmbedding',
    'Identity',
    'FairseqAttention',
    'residual',
    'NLCBatchNorm1d', 'LayerNorm', 'MyDropout', 'Residual',
]
