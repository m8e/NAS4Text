#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th

__author__ = 'fyabc'


def combine_outputs(op, output_list, **kwargs):
    op = op.lower()
    if op == 'add':
        return th.stack([t for t in output_list if t is not None]).mean(dim=0)
    elif op == 'concat':
        reduce_op = kwargs.pop('reduce_op', None)
        if reduce_op is None:
            raise ValueError('The kwarg "reduce_op" is required for "concat" combine op')
        return reduce_op(th.cat([t for t in output_list if t is not None], dim=-1).transpose(1, 2)).transpose(1, 2)
    elif op == 'last':
        return output_list[-1]
    else:
        raise ValueError('Unknown block combine op {!r}'.format(op))
