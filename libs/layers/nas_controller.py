#! /usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'fyabc'


class NASController:
    def __init__(self, net_code, hparams):
        self.hparams = hparams
        self.net_code = net_code

    def get_weight(self, index, input_index, op_type, **kwargs):
        raise NotImplementedError()


__all__ = [
    'NASController',
]
