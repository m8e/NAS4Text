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
    # TODO: Discuss on it (BN on RNN?)
    def forward(self, input_):
        if input_.data.ndimension() == 3:
            return super().forward(input_.transpose(1, 2)).transpose(1, 2)
        return super().forward(input_)


def _push_processors(hparams, module_list: nn.ModuleList, ops):
    for op in ops:
        if op == PPPSpace.Dropout:
            module_list.append(nn.Dropout(p=hparams.dropout))
        elif op == PPPSpace.Norm:
            # TODO
            norm_type = hparams.norm_type
            if norm_type == 'layer':
                pass
            elif norm_type == 'batch':
                pass
            elif norm_type == 'noam':
                pass
            elif norm_type == 'none':
                pass
            else:
                raise ValueError("Unknown normalization type {}".format(norm_type))
        else:
            raise ValueError('Unknown op {}'.format(op))


def push_prepostprocessors(layer, preprocess_code, postprocess_code):
    _push_processors(layer.hparams, layer.preprocessors, PPPSpace.get_ops(preprocess_code))
    _push_processors(layer.hparams, layer.postprocessors, PPPSpace.get_ops(postprocess_code))
