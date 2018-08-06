#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th

__author__ = 'fyabc'


def combine_outputs(op, output_list, **kwargs):
    op = op.lower()
    if op == 'add':
        return th.stack([t for t in output_list if t is not None]).mean(dim=0)
    elif op == 'concat':
        linear = kwargs.pop('linear', None)
        if linear is None:
            raise ValueError('The kwarg "linear" is required for "concat" combine op')
        return linear(th.cat([t for t in output_list if t is not None], dim=-1))
    elif op == 'last':
        return output_list[-1]
    else:
        raise ValueError('Unknown block combine op {!r}'.format(op))
