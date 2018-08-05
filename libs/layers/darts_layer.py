#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn

from .base import ChildLayer, wrap_ppp
from .block_node_ops import BlockNodeOp
from ..utils.search_space import DartsSpace
from ..utils import functions as fns
from ..layers.ppp import push_prepostprocessors

__author__ = 'fyabc'


class DartsMixedOp(nn.Module):
    def __init__(self, hparams, input_shape, in_encoder=True, **kwargs):
        super().__init__()
        self.hparams = hparams
        self.in_encoder = in_encoder

        self.ops = nn.ModuleList()

        # Build all ops.
        supported_ops = self.supported_ops(self.in_encoder)
        for op_type, op_args in supported_ops.values():
            self.ops.append(op_type(op_args, input_shape, hparams=hparams, in_encoder=in_encoder, **kwargs))

    @staticmethod
    def supported_ops(in_encoder=True):
        """Get supported ops and related op args. Only support a subset of block ops."""
        darts_ops = DartsSpace.CellOps
        result = {
            k: (v, darts_ops[k])
            for k, v in BlockNodeOp.supported_ops().items()
            if k in darts_ops
        }
        if in_encoder:
            result.pop('EncoderAttention')
        return result

    def forward(self, x, weights, lengths=None, encoder_state=None, **kwargs):
        return sum(w * op(x, lengths=lengths, encoder_state=encoder_state, **kwargs)
                   for w, op in zip(weights, self.ops))


class DartsLayer(ChildLayer):
    """Darts layer, aka a 'Cell'. Include a DAG of nodes in it.

    [NOTE]: Nodes and edges
        Node 0, 1, ... I - 1: inputs (default I = 2)
        Node I, I + 1, ..., I + N - 1: internal nodes (default N = 4)
        Edge i -> j: mixed op
    """

    def __init__(self, hparams, input_shape, in_encoder=True):
        super().__init__(hparams)
        self.in_encoder = in_encoder
        self.num_nodes = hparams.num_nodes
        self.num_input_nodes = 2

        # [NOTE]: Last N nodes will be combined to the block output.
        self.num_output_nodes = hparams.num_output_nodes

        # FIXME: Some hyperparameters are fixed now.

        self.node_combine_op = 'add'
        self.block_combine_op = hparams.block_combine_op.lower()

        assert self.num_input_nodes == 2, 'Number of input nodes != 2 is not supported now'

        # [NOTE]: Does NOT use residual in block ppp and node ppp.
        self.ppp_code = ['', '']
        push_prepostprocessors(self, self.ppp_code[0], self.ppp_code[1], input_shape, input_shape)

        self.node_ppp_code = ['', 'n']  # PPP of each node.
        # The list of empty layers to store ppp.
        self.node_ppp_list = nn.ModuleList()

        self.mixed_ops = nn.ModuleList()
        self.offsets = {}   # Map edge to offset.
        self._build_nodes(input_shape)

    @staticmethod
    def supported_ops(in_encoder=True):
        return DartsMixedOp.supported_ops(in_encoder)

    @property
    def num_total_nodes(self):
        return self.num_input_nodes + self.num_nodes

    def _build_nodes(self, input_shape):
        offset = 0
        for j in range(self.num_input_nodes, self.num_total_nodes):
            # Node ppp
            node_ppp = ChildLayer(self.hparams)
            push_prepostprocessors(node_ppp, self.node_ppp_code[0], self.node_ppp_code[1], input_shape, input_shape)
            self.node_ppp_list.append(node_ppp)

            # Edges
            for i in range(j):
                # Mixed op of edge i -> j
                self.mixed_ops.append(DartsMixedOp(
                    self.hparams, input_shape, in_encoder=self.in_encoder,
                    controller=None, index=j, input_index=i,
                ))
                self.offsets[(i, j)] = offset
                offset += 1

    @wrap_ppp(2)
    def forward(self, input_, prev_input, weights, lengths=None, encoder_state=None, **kwargs):
        """

        Args:
            input_:
            prev_input:
            weights: Alpha weights of all output.
            lengths:
            encoder_state:
            **kwargs:

        Returns:

        """

        node_output_list = [None for _ in range(self.num_total_nodes)]
        node_output_list[0] = input_
        node_output_list[1] = prev_input

        for j in range(self.num_input_nodes, self.num_total_nodes):
            node_ppp = self.node_ppp_list[j - self.num_input_nodes]
            results = []
            for i in range(j):
                offset = self.offsets[(i, j)]
                mixed_op = self.mixed_ops[offset]
                mixed_op_weights = weights[offset]
                edge_input = node_ppp.preprocess(node_output_list[i])
                results.append(mixed_op(
                    edge_input, mixed_op_weights,
                    lengths=lengths, encoder_state=encoder_state, **kwargs,
                ))
            node_output = fns.combine_outputs(self.node_combine_op, results, linear=None)
            # [NOTE]: Only use first node to compute postprocess, usually omit it (does NOT use residual).
            node_output_list[j] = node_ppp.postprocess(node_output, node_output_list[0])
        return fns.combine_outputs(self.block_combine_op, node_output_list[-self.num_output_nodes:], linear=None)
