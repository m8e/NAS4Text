#! /usr/bin/python
# -*- coding: utf-8 -*-

"""The base class of child layers."""

import torch.nn as nn

from . import ppp

__author__ = 'fyabc'


class ChildLayer(nn.Module):
    def __init__(self, hparams, preprocess_code, postprocess_code):
        super().__init__()
        self.hparams = hparams

        self.preprocessors = nn.ModuleList()
        self.postprocessors = nn.ModuleList()

        ppp.push_prepostprocessors(self, preprocess_code, postprocess_code)

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
