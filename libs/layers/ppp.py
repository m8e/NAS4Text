#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Pre/post processing of layers."""

import torch.nn as nn

from .common import NLCBatchNorm1d, LayerNorm, MyDropout, Residual
from ..utils.search_space import PPPSpace

__author__ = 'fyabc'


def _push_processors(hparams, module_list: nn.ModuleList, ops, shape):
    for op in ops:
        if op == PPPSpace.Dropout:
            module_list.append(MyDropout(p=hparams.ppp_dropout))
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
        elif op == PPPSpace.Residual:
            module_list.append(Residual())
        else:
            raise ValueError('Unknown op {}'.format(op))


def push_prepostprocessors(layer, preprocess_code, postprocess_code, input_shape, output_shape):
    pre_ops = PPPSpace.get_ops(preprocess_code)
    assert PPPSpace.Residual not in pre_ops, 'Residual connection cannot be preprocessor'
    _push_processors(layer.hparams, layer.preprocessors, pre_ops, input_shape)

    post_ops = PPPSpace.get_ops(postprocess_code)
    _push_processors(layer.hparams, layer.postprocessors, post_ops, output_shape)
    if PPPSpace.Residual in post_ops:
        layer.build_residual_projection(input_shape, output_shape)
