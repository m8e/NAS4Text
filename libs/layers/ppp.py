#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Pre/post processing of layers."""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

__author__ = 'fyabc'


class PPPSpace:
    Dropout = 'dropout'
    Norm = 'norm'

    # None, Dropout, Norm, Dropout + Norm, Norm + Dropout
    Preprocessors = [0, 1, 2, 3, 4]

    # None, Dropout, Norm, Dropout + Norm, Norm + Dropout
    Postprocessors = [0, 1, 2, 3, 4]

    @classmethod
    def get_ops(cls, code):
        if code == 0:
            return []
        elif code == 1:
            return [cls.Dropout]
        elif code == 2:
            return [cls.Norm]
        elif code == 3:
            return [cls.Dropout, cls.Norm]
        elif code == 4:
            return [cls.Norm, cls.Dropout]
        elif code is None:
            return []
        else:
            raise ValueError('Unknown code {}'.format(code))


class NLCBatchNorm1d(nn.BatchNorm1d):
    """Batch normalization, applied on (N, L, C) input."""
    # TODO: Discuss on it (BN on RNN?), need test
    def forward(self, input_):
        if input_.data.ndimension() == 3:
            return super().forward(input_.transpose(1, 2)).transpose(1, 2)
        return super().forward(input_)


class LayerNorm(nn.Module):
    """A Simple implementation of layer normalization, applied on (N, L, C) input."""

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.gamma = nn.Parameter(th.ones(num_features))
        self.beta = nn.Parameter(th.zeros(num_features))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def __repr__(self):
        return '{name}({num_features}, eps={eps})'.format(name=self.__class__.__name__, **self.__dict__)


def _push_processors(hparams, module_list: nn.ModuleList, ops, shape):
    for op in ops:
        if op == PPPSpace.Dropout:
            module_list.append(nn.Dropout(p=hparams.dropout))
        elif op == PPPSpace.Norm:
            norm_type = hparams.norm_type
            if norm_type == 'layer':
                module_list.append(LayerNorm(shape[-1], hparams.norm_epsilon))
            elif norm_type == 'batch':
                # TODO: Need test
                module_list.append(NLCBatchNorm1d(shape[-1], hparams.norm_epsilon))
            elif norm_type == 'noam':
                # TODO:
                pass
            elif norm_type == 'none':
                # do nothing
                pass
            else:
                raise ValueError("Unknown normalization type {}".format(norm_type))
        else:
            raise ValueError('Unknown op {}'.format(op))


def push_prepostprocessors(layer, preprocess_code, postprocess_code, input_shape, output_shape):
    _push_processors(layer.hparams, layer.preprocessors, PPPSpace.get_ops(preprocess_code), input_shape)
    _push_processors(layer.hparams, layer.postprocessors, PPPSpace.get_ops(postprocess_code), output_shape)
