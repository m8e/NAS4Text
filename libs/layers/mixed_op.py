#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn

from .block_node_ops import BlockNodeOp
from ..utils.search_space import SearchSpace

__author__ = 'fyabc'


class MixedOp(nn.Module):
    _SupportedOps = {}  # [NOTE]: Each subclass should create its own ``_SupportedOps``.
    _Space = SearchSpace  # [NOTE]: Subclasses can set different space.

    def __init__(self, hparams, input_shape, in_encoder=True, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.in_encoder = in_encoder

        self.ops = nn.ModuleList()

        # Build all ops.
        supported_ops = self.supported_ops()
        for op_name, op_type, op_args in supported_ops:
            self.ops.append(op_type(op_args, input_shape, hparams=hparams, in_encoder=in_encoder, **kwargs))

    def supported_ops(self):
        """Get supported ops and related op args. Only support a subset of block ops.

        Returns:
            list: Each item is [op_name, op_type, op_args].
        """

        in_encoder = self.in_encoder

        name2ops = BlockNodeOp.supported_ops()

        result = self._SupportedOps.get(in_encoder, None)
        if result is None:
            result = self._Space.CellOpSpaces[self.hparams.cell_op_space]

            def bad_condition(o):
                if in_encoder:
                    return o[0] == 'EncoderAttention'
                else:
                    return o[0] == 'LSTM' and o[1][1] is True

            result = [[o[0], name2ops[o[0]], o[1]] for o in result if not bad_condition(o)]
            self._SupportedOps[in_encoder] = result
        return result

    def forward(self, x, weights, lengths=None, encoder_state=None, **kwargs):
        raise NotImplementedError()
