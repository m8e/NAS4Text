#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections.abc import Sequence, Mapping

import torch as th
import torch.nn as nn

from .base import ChildLayer, wrap_ppp
from .ppp import push_prepostprocessors
from .block_node_ops import BlockNodeOp, BlockCombineNodeOp
from ..utils.search_space import CellSpace, PPPSpace
from ..layers.common import Linear

__author__ = 'fyabc'


class InputNode(nn.Module):
    """Input node."""

    def __init__(self, input_index):
        super().__init__()
        self.input_index = input_index

        assert self.input_index in (0, 1)

    def forward(self, in1, in2, lengths=None, encoder_state=None):
        """

        :param in1: Input
        :param in2: Previous input. None for the first layer.
        :param lengths:
        :param encoder_state:
        :return:
        """
        result = (in1, in2)[self.input_index]
        if result is None:
            assert self.input_index == 1
            result = in1
        return result

    def contains_lstm(self):
        return False


class Node(nn.Module):
    """Block node.

    Input[1], Input[2], Op[1], Op[2], CombineOp => Output
    """

    def __init__(self, in1, in2, op1, op2, combine_op, input_shape, in_encoder=True, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.in_encoder = in_encoder
        self.in1_index = in1
        self.in2_index = in2
        self.op1 = self._parse_op(op1, input_shape)
        self.op2 = self._parse_op(op2, input_shape)
        self.combine_op = self._parse_combine_op(combine_op, input_shape)

    @staticmethod
    def _normalize_op_code(op_code, space=CellSpace.CellOps):
        if isinstance(op_code, Sequence) and not isinstance(op_code, str):
            # [op_code, op_arg1, op_arg2, ...]
            op_code, *op_args = op_code
        else:
            op_args = tuple()

        if isinstance(op_code, str):
            op_code = space[op_code]
        assert isinstance(op_code, int)

        return op_code, op_args

    def _parse_op(self, op_code, input_shape):
        op_code, op_args = self._normalize_op_code(op_code)
        return BlockNodeOp.create(op_code, op_args, input_shape, self.in_encoder, hparams=self.hparams)

    def _parse_combine_op(self, op_code, input_shape):
        op_code, op_args = self._normalize_op_code(op_code, space=CellSpace.CombineOps)
        return BlockCombineNodeOp.create(op_code, op_args, input_shape, self.in_encoder, hparams=self.hparams)

    def forward(self, in1, in2, lengths=None, encoder_state=None, **kwargs):
        in1, in2 = self._process_inputs(in1, in2)
        return self.combine_op(
            self.op1(in1, lengths=lengths, encoder_state=encoder_state, **kwargs),
            self.op2(in2, lengths=lengths, encoder_state=encoder_state, **kwargs),
            lengths=lengths,
            encoder_state=encoder_state,
        )

    def _process_inputs(self, in1, in2):
        # TODO: More processing.
        return in1, in2

    def forward_in_block(self, node_output_list, lengths=None, encoder_state=None, **kwargs):
        return self(node_output_list[self.in1_index], node_output_list[self.in2_index],
                    lengths=lengths, encoder_state=encoder_state, **kwargs)

    def contains_lstm(self):
        from .block_node_ops import LSTMOp
        return isinstance(self.op1, LSTMOp) or isinstance(self.op2, LSTMOp)


class CombineNode(nn.Module):
    """Combine node."""

    def __init__(self, in_encoder=True, hparams=None):
        super().__init__()
        self.in_encoder = in_encoder
        self.hparams = hparams
        self.op = hparams.block_combine_op.lower()
        assert self.op in ('add', 'concat'), 'Unknown block combine op {!r}'.format(self.op)

        self.linear = None

    def set_concat_linear(self, hidden_size, n):
        if self.op == 'concat':
            self.linear = Linear(
                hidden_size * n, hidden_size, bias=True, dropout=self.hparams.dropout, hparams=self.hparams)

    def forward(self, node_output_list):
        if self.op == 'add':
            return th.stack([t for t in node_output_list if t is not None]).mean(dim=0)
        else:
            assert self.op == 'concat'
            return self.linear(th.cat([t for t in node_output_list if t is not None], dim=-1))


class BlockLayer(ChildLayer):
    """Block layer. Contains several nodes."""

    def __init__(self, hparams, in_encoder=True):
        super().__init__(hparams)
        self.in_encoder = in_encoder
        self.input_node_indices = []
        self.nodes = nn.ModuleList()
        self.combine_node = CombineNode(in_encoder=in_encoder, hparams=hparams)
        self.topological_order = []
        self.block_params = {}

    def build(self, layer_code, input_shape):
        if isinstance(layer_code[-1], Mapping):
            # The last item of the block code can be block params.
            self.block_params = dict(layer_code[-1])
            layer_code = layer_code[:-1]
        self._get_topological_order(layer_code)

        for i, node_code in enumerate(layer_code):
            in1, in2, op1, op2, combine_op = node_code
            if in1 is None:
                # This is an input node.
                self.nodes.append(InputNode(input_index=len(self.input_node_indices)))
                self.input_node_indices.append(i)
            else:
                # This is a normal node.
                self.nodes.append(Node(in1, in2, op1, op2, combine_op, input_shape,
                                       in_encoder=self.in_encoder, hparams=self.hparams))

        if len(self.input_node_indices) != 2:
            raise RuntimeError('The block layer must have exactly two input nodes, but got {}'.format(
                len(self.input_node_indices)))
        self.combine_node.set_concat_linear(input_shape[-1], n=len(self.nodes) - len(self.input_node_indices))

        push_prepostprocessors(
            self, self.block_params.get('preprocessors', self.hparams.default_block_preprocessors),
            self.block_params.get('postprocessors', self.hparams.default_block_postprocessors),
            input_shape, input_shape,
        )

        return input_shape

    @wrap_ppp
    def forward(self, input_, prev_input, lengths=None, encoder_state=None, **kwargs):
        """

        :param input_:
        :param prev_input:
        :param lengths:
        :param encoder_state:
        :return:
        """

        node_output_list = [None for _ in self.nodes]

        for i in self.topological_order:
            node = self.nodes[i]
            if isinstance(node, InputNode):
                node_output_list[i] = node(input_, prev_input)
            else:
                assert isinstance(node, Node)
                node_output_list[i] = node.forward_in_block(
                    node_output_list, lengths=lengths, encoder_state=encoder_state, **kwargs)

        # Combine all intermediate nodes, does not combine input nodes.
        return self.combine_node([o for i, o in enumerate(node_output_list) if i not in self.input_node_indices])

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

    def contains_lstm(self):
        return any(n.contains_lstm() for n in self.nodes)


def build_block(layer_code, input_shape, hparams, in_encoder=True):
    """

    Args:
        layer_code:
        input_shape: torch.Size object
            Shape of input tensor, expect (batch_size, seq_len, input_size)
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
