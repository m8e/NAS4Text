#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Common used layers."""

import copy
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .learned_positional_embedding import LearnedPositionalEmbedding
from .sinusoidal_positional_embedding import SinusoidalPositionalEmbedding

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
    elif hparams.initializer == 'fairseq':
        nn.init.xavier_uniform_(m.weight)
        if bias:
            nn.init.constant_(m.bias, 0.)
        return m
    else:
        raise ValueError('Unknown initializer {!r}'.format(hparams.initializer))


def Embedding(num_embeddings, embedding_dim, padding_idx, hparams=None):
    """Weight-normalized Embedding layer"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    if hparams.initializer in ('original', 'kaitao', 'kaitao_wn'):
        m.weight.data.normal_(0, 0.1)
    elif hparams.initializer == 'uniform_unit_scaling':
        uniform_unit_scaling_initializer(m.weight, scale=hparams.initializer_gain)
    elif hparams.initializer == 'fairseq':
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        raise ValueError('Unknown initializer {!r}'.format(hparams.initializer))
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, hparams=None, learned=True):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        if hparams.initializer in ('original', 'kaitao', 'kaitao_wn'):
            m.weight.data.normal_(0, 0.1)
        elif hparams.initializer == 'uniform_unit_scaling':
            uniform_unit_scaling_initializer(m.weight, scale=hparams.initializer_gain)
        elif hparams.initializer == 'fairseq':
            nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
            nn.init.constant_(m.weight[padding_idx], 0)
        else:
            raise ValueError('Unknown initializer {!r}'.format(hparams.initializer))
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m


class Identity(nn.Module):
    def forward(self, x):
        return x


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


class _LayerNorm(nn.Module):
    """A Simple implementation of layer normalization, applied on (N, L, C) input."""

    def __init__(self, num_features, eps=1e-5):
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


if hasattr(nn, 'LayerNorm'):
    class _FairseqLayerNorm(nn.LayerNorm):
        """Wrapper of nn.LayerNorm, make the forward arguments compatible."""

        def forward(self, x, input_=None):
            return super().forward(x)
else:
    _FairseqLayerNorm = None


def LayerNorm(*args, **kwargs):
    if _FairseqLayerNorm is not None:
        return _FairseqLayerNorm(*args, **kwargs)
    return _LayerNorm(*args, **kwargs)


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
    'residual',
    'NLCBatchNorm1d', 'LayerNorm', 'MyDropout', 'Residual',
]
