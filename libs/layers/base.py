#! /usr/bin/python
# -*- coding: utf-8 -*-

"""The base class of child layers."""

import torch.nn as nn

__author__ = 'fyabc'


class ChildLayer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.preprocessors = nn.ModuleList()
        self.postprocessors = nn.ModuleList()

    def forward(self, *args):
        raise NotImplementedError

    def preprocess(self, x):
        for m in self.preprocessors:
            x = m(x)
        return x

    def postprocess(self, x):
        for m in self.postprocessors:
            x = m(x)
        return x
