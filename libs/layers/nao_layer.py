#! /usr/bin/python
# -*- coding: utf-8 -*-

from .base import ChildLayer

__author__ = 'fyabc'


class NAOLayer(ChildLayer):
    def __init__(self, hparams, input_shape, in_encoder=True):
        super().__init__(hparams)
        self.in_encoder = in_encoder
        self.num_nodes = hparams.num_nodes
        self.num_input_nodes = 2

        # TODO

    def forward(self, *input):
        raise RuntimeError('This method must not be called')
