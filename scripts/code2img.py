#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate image from net code."""

import argparse
from collections import Sequence
import os
import sys

import graphviz as gv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.layers.net_code import load_net_code_from_file, NetCode
from libs.utils.search_space import CellSpace


def _name(block_name, i, s=None):
    if s is None:
        return '{}.{}'.format(block_name, i)
    return '{}.{}.{}'.format(block_name, i, s)


def _int2str(i, space=CellSpace.CellOps):
    if isinstance(i, str):
        return i
    for k, v in space.items():
        if v == i:
            return k
    print('#', i)


def _split_op_args(op_code):
    if isinstance(op_code, Sequence) and not isinstance(op_code, str):
        # [op_code, op_arg1, op_arg2, ...]
        op_code, *op_args = op_code
    else:
        op_args = tuple()
    return op_code, op_args


def _make_cell_subgraph(block_name, i, in1, in2, op1, op2, combine_op):
    c = gv.Digraph(name='cluster_{}_{}'.format(block_name, i))
    c.graph_attr.update({
        'label': 'cell {}'.format(i),
        'labelloc': 't',
    })
    c.node_attr.update({
        'shape': 'box',
        'style': 'filled',
    })

    i1_n = _name(block_name, i, 'in1')
    i2_n = _name(block_name, i, 'in2')
    op1_n = _name(block_name, i, 'op1')
    op2_n = _name(block_name, i, 'op2')
    combine_op_n = _name(block_name, i, 'combine')

    c.node(i1_n, 'in1', fillcolor='lightblue')
    c.node(i2_n, 'in2', fillcolor='lightblue')
    c.node(op1_n, _int2str(op1), fillcolor='green')
    c.node(op2_n, _int2str(op2), fillcolor='green')
    c.node(combine_op_n, _int2str(combine_op, space=CellSpace.CombineOps), fillcolor='orange')
    c.edge(i1_n, op1_n)
    c.edge(i2_n, op2_n)
    c.edge(op1_n, combine_op_n)
    c.edge(op2_n, combine_op_n)

    return c


def main(args=None):
    parser = argparse.ArgumentParser(description='Generate image from net code.')
    parser.add_argument('file', help='Net code file')
    parser.add_argument('-T', '--format', default='jpg', help='Output format, default is %(default)r')
    parser.add_argument('-d', '--dir', help='Output directory', required=True)
    parser.add_argument('-o', '--output', help='Output filename (without format ext)', default=None)

    args = parser.parse_args(args)
    print(args)

    code_file = args.file
    output_basename = os.path.basename(code_file) if args.output is None else args.output
    output_file = os.path.join(args.dir, output_basename)

    os.makedirs(args.dir, exist_ok=True)

    code = load_net_code_from_file(code_file)

    if code.type == NetCode.Default:
        raise NotImplementedError()
    elif code.type == NetCode.BlockChildNet:
        blocks = code.blocks
    else:
        raise RuntimeError('The net code type {!r} is not supported yet'.format(code.type))

    g_global = gv.Digraph()

    g_global.format = args.format
    g_global.name = 'G'

    for name, block in blocks.items():
        g = gv.Digraph(name='cluster_' + name)
        g.graph_attr.update({
            'label': name,
            'labelloc': 't',
        })
        g.node_attr.update({
            'shape': 'box',
            'style': 'filled',
        })

        input_node_indices = []
        for i, cell in enumerate(block):
            node_name = _name(name, i)
            in1, in2, op1, op2, combine_op = cell

            op1, op_args1 = _split_op_args(op1)
            op2, op_args2 = _split_op_args(op2)
            combine_op, combine_op_args = _split_op_args(combine_op)

            if in1 is None:
                g.node(node_name, 'input {}'.format(len(input_node_indices)))
                input_node_indices.append(i)
            else:
                g.subgraph(_make_cell_subgraph(name, i, in1, in2, op1, op2, combine_op))

                # Add input edges.
                for in_, in_name in zip((in1, in2), ('in1', 'in2')):
                    if in_ in input_node_indices:
                        g.edge(_name(name, in_), _name(name, i, in_name))
                    else:
                        g.edge(_name(name, in_, 'combine'), _name(name, i, in_name))

        g_global.subgraph(g)

    print(g_global.source)

    print(g_global.render(filename=output_file, cleanup=True))


if __name__ == '__main__':
    main()
