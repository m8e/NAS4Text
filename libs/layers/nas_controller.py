#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'fyabc'


class NASController:
    def __init__(self, hparams):
        self.hparams = hparams

    def get_weight(self, in_encoder, layer_id, index, input_index, op_code, **kwargs):
        raise NotImplementedError()

    def get_combine_weight(self, in_encoder, layer_id, index, op_code, **kwargs):
        raise NotImplementedError()


__all__ = [
    'NASController',
]
