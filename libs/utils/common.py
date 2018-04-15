#! /usr/bin/python
# -*- coding: utf-8 -*-

import contextlib

import numpy as np
import torch as th
from torch.autograd import Variable

__author__ = 'fyabc'


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)
    if not hasattr(make_positions, 'range_buf'):
        make_positions.range_buf = tensor.new()
    make_positions.range_buf = make_positions.range_buf.type_as(tensor)
    if make_positions.range_buf.numel() < max_pos:
        th.arange(padding_idx + 1, max_pos, out=make_positions.range_buf)
    mask = tensor.ne(padding_idx)
    positions = make_positions.range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


def mask_from_lengths(lengths, left_pad):
    if isinstance(lengths, Variable):
        lengths_ = lengths.data
    else:
        lengths_ = lengths
    batch_size = len(lengths_)
    max_length = max(lengths_)
    result = th.ByteTensor(batch_size, max_length).fill_(0)
    if isinstance(lengths, Variable):
        result = Variable(result, requires_grad=lengths.requires_grad)

    for i, length in enumerate(lengths_):
        if left_pad:
            result[i, max_length - length:] = 1
        else:
            result[i, :length] = 1

    return result
