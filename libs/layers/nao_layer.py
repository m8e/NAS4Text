#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn

from .base import ChildLayer
from .mixed_op import MixedOp
from ..layers.ppp import push_prepostprocessors

__author__ = 'fyabc'


class NAOLayer(ChildLayer):
    def __init__(self, hparams, input_shape, in_encoder=True):
        super().__init__(hparams)
        self.in_encoder = in_encoder
        self.num_nodes = hparams.num_nodes
        self.num_input_nodes = 2

        # FIXME: Some hyperparameters are fixed now.

        assert self.num_input_nodes == 2, 'Number of input nodes != 2 is not supported now'

        # [NOTE]: Does NOT use residual in block ppp and node ppp.
        self.ppp_code = ['', '']
        push_prepostprocessors(self, self.ppp_code[0], self.ppp_code[1], input_shape, input_shape)

        self.node_ppp_code = ['', 'n']  # PPP of each node.
        # The list of empty layers to store ppp.
        self.node_ppp_list = nn.ModuleList()

        self.edges = nn.ModuleList()
        self.offsets = {}  # Map edge to offset.
        self._build_nodes(input_shape)

    def supported_ops(self):
        return self.edges[0].supported_ops()

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
                self.edges.append(MixedOp(
                    self.hparams, input_shape, in_encoder=self.in_encoder,
                    controller=None, index=j, input_index=i,
                ))
                self.offsets[(i, j)] = offset
                offset += 1

    def forward(self, *input):
        raise RuntimeError('This method must not be called')
