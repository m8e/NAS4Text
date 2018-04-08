#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convolutional layer.

Layer code:
[CNN, KernelSize, ...]
"""

import torch.nn as nn

__author__ = 'fyabc'


def build_cnn(layer_code, input_shape, hparams):
    """

    Args:
        layer_code:
        input_shape: torch.Size object
            Shape of input tensor, expect (batch_size, seq_len, input_size)
        hparams:

    Returns: layer, output_shape
    """

    raise NotImplementedError('Convolutional layer not implemented')
