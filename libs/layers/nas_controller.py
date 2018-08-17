#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'fyabc'


class NASController:
    def __init__(self, hparams):
        self.hparams = hparams

    def get_weight(self, in_encoder, layer_id, index, input_index, op_type, **kwargs):
        raise NotImplementedError()


__all__ = [
    'NASController',
]
