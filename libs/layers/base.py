#! /usr/bin/python
# -*- coding: utf-8 -*-

"""The base class of child layers."""

import torch.nn as nn

from .common import Linear

__author__ = 'fyabc'


class ChildLayer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # The projection for residual connection (only for different input/output dim).
        self.residual_projection = None
        self.preprocessors = nn.ModuleList()
        self.postprocessors = nn.ModuleList()

    def forward(self, *args):
        raise NotImplementedError

    def build_residual_projection(self, input_shape, output_shape):
        self.residual_projection = Linear(input_shape[2], output_shape[2], hparams=self.hparams) \
            if input_shape != output_shape else None

    def preprocess(self, x):
        for m in self.preprocessors:
            x = m(x)
        return x

    def postprocess(self, x, input_):
        input_ = self.modify_input_before_postprocess(input_)
        for m in self.postprocessors:
            x = m(x, input_)
        return x

    def modify_input_before_postprocess(self, input_):
        if self.residual_projection is not None:
            input_ = self.residual_projection(input_)
        return input_
