#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections.abc import Sequence

import torch as th
import torch.nn as nn

from .base import ChildLayer
from ..utils.search_space import CellSpace

__author__ = 'fyabc'


class InputNode(nn.Module):
    """Input node."""

    def __init__(self, input_index):
        super().__init__()
        self.input_index = input_index

    def forward(self, in1, in2, lengths=None, encoder_state=None):
        return (in1, in2)[self.input_index]


class Node(nn.Module):
    """Block node.

    Input[1], Input[2], Op[1], Op[2], CombineOp => Output
    """

    def __init__(self, in1, in2, op1, op2, combine_op, input_shape, in_encoder=True):
        super().__init__()
        self.in_encoder = in_encoder
        self.in1_index = in1
        self.in2_index = in2
        self.op1 = self._parse_op(op1, input_shape)
        self.op2 = self._parse_op(op2, input_shape)
        self.combine_op = self._parse_combine_op(combine_op, input_shape)

    def _parse_op(self, op_code, input_shape):
        if isinstance(op_code, Sequence):
            # [op_code, op_arg1, op_arg2, ...]
            op_code, *op_args = op_code
        else:
            op_args = tuple()

        if isinstance(op_code, str):
            op_code = getattr(CellSpace, op_code)
        assert isinstance(op_code, int)

        if op_code == CellSpace.LSTM:
            raise NotImplementedError()
        elif op_code == CellSpace.R2L_LSTM:
            raise NotImplementedError()
        elif op_code == CellSpace.CNN:
            raise NotImplementedError()
        elif op_code == CellSpace.SelfAttention:
            raise NotImplementedError()
        elif op_code == CellSpace.GroupedCNN:
            raise NotImplementedError()
        elif op_code == CellSpace.FFN:
            raise NotImplementedError()
        elif op_code == CellSpace.Identity:
            raise NotImplementedError()
        elif op_code == CellSpace.GroupedLSTM:
            raise NotImplementedError()
        else:
            raise RuntimeError('Unknown op code {}'.format(op_code))

    def _parse_combine_op(self, op_code, input_shape):
        if op_code == CellSpace.Add:
            raise NotImplementedError()
        elif op_code == CellSpace.Concat:
            raise NotImplementedError()
        else:
            raise RuntimeError('Unknown combine op code {}'.format(op_code))

    def forward(self, in1, in2, lengths=None, encoder_state=None):
        # TODO: Process the situation that in2 is None.
        if self.in_encoder:
            return self.combine_op(self.op1(in1), self.op2(in2))
        else:
            raise NotImplementedError('Decoder node not implemented')

    def forward_in_block(self, node_output_list, lengths=None, encoder_state=None):
        return self(node_output_list[self.in1_index], node_output_list[self.in2_index],
                    lengths=lengths, encoder_state=encoder_state)


class CombineNode(nn.Module):
    """Combine node."""

    def forward(self, node_output_list):
        # TODO: Combine outputs
        pass


class BlockLayer(ChildLayer):
    """Block layer. Contains several nodes."""

    def __init__(self, hparams, in_encoder=True):
        super().__init__(hparams)
        self.in_encoder = in_encoder
        self.input_node_indices = []
        self.nodes = nn.ModuleList()
        self.combine_node = CombineNode()
        self.topological_order = []

    def build(self, layer_code, input_shape):
        self._get_topological_order(layer_code)

        for i, node_code in enumerate(layer_code):
            in1, in2, op1, op2, combine_op = node_code
            if in1 is None:
                # This is an input node.
                self.nodes.append(InputNode(input_index=len(self.input_node_indices)))
                self.input_node_indices.append(i)
            else:
                # This is a normal node.
                self.nodes.append(Node(in1, in2, op1, op2, combine_op, input_shape, in_encoder=self.in_encoder))

        if len(self.input_node_indices) != 2:
            raise RuntimeError('The block layer must have exactly two input nodes, but got {}'.format(
                len(self.input_node_indices)))
        return input_shape

    def forward(self, input_, prev_input, lengths=None, encoder_state=None):
        node_output_list = [None for _ in self.nodes]

        for i in self.topological_order:
            node = self.nodes[i]
            if isinstance(node, InputNode):
                node_output_list[i] = node(input_, prev_input)
            else:
                assert isinstance(node, Node)
                node_output_list[i] = node.forward_in_block(
                    node_output_list, lengths=lengths, encoder_state=encoder_state)

        return self.combine_node(node_output_list)

    def _get_topological_order(self, layer_code):
        self.topological_order = []
        remain_nodes = list(range(len(layer_code)))

        while remain_nodes:
            new_remain_nodes = []
            for i in remain_nodes:
                in1, in2, *_ = layer_code[i]

                if in1 is None:
                    # This is an input node.
                    self.topological_order.append(i)
                else:
                    # This is a normal node.
                    if in1 in self.topological_order and in2 in self.topological_order:
                        self.topological_order.append(i)
                    else:
                        new_remain_nodes.append(i)
            remain_nodes = new_remain_nodes


def build_block(layer_code, input_shape, hparams, in_encoder=True):
    """

    Args:
        layer_code:
        input_shape:
        hparams:
        in_encoder:

    Returns:
        tuple
    """
    block = BlockLayer(hparams, in_encoder)
    output_shape = block.build(layer_code, input_shape)

    return block, output_shape


__all__ = [
    'BlockLayer',
    'build_block',
]
