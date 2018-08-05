#! /usr/bin/python
# -*- coding: utf-8 -*-

"""The base class of child layers."""

import functools

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
        if self.preprocessors is None:
            return x
        for m in self.preprocessors:
            x = m(x)
        return x

    def postprocess(self, x, input_):
        if self.postprocessors is None:
            return x
        input_ = self.modify_input_before_postprocess(input_)
        for m in self.postprocessors:   # [DEBUG]: Same after here (except dropout)
            x = m(x, input_)
        return x

    def modify_input_before_postprocess(self, input_):
        if self.residual_projection is not None:
            input_ = self.residual_projection(input_)
        return input_

    def simplify(self):
        self.preprocessors = None
        self.postprocessors = None
        return self


def wrap_ppp(forward_fn_or_i):
    """Wrap the forward function with ppp.

    Usage:
        class C(nn.Module):
            @wrap_ppp
            def forward(self, input_, a, b, c):
                pass

        # Multiple inputs (e.g. residual connection)
        class C(nn.Module):
            @wrap_ppp(2)
            def forward(self, input_, prev_input, a, b, c):
                pass

    [NOTE]: When use multiple input, All inputs will be preprocessed by same preprocessors,
        But only the first input will be used in postprocessors.
    """
    def wrapper(forward_fn):
        @functools.wraps(forward_fn)
        def forward(self, *args, **kwargs):
            args = list(args)
            first_input_before = args[0]
            for i in range(num_inputs):
                args[i] = self.preprocess(args[i])
            result = forward_fn(self, *args, **kwargs)
            return self.postprocess(result, first_input_before)

        return forward

    if callable(forward_fn_or_i):
        num_inputs = 1
        return wrapper(forward_fn_or_i)
    else:
        assert isinstance(forward_fn_or_i, int), 'Must pass an int to the decorator'
        num_inputs = forward_fn_or_i
        return wrapper
