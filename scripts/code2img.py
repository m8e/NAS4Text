#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate image from net code."""

import argparse
from collections import Sequence
import json
import os
import sys

import graphviz as gv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.layers.net_code import load_net_code_from_file, NetCode
import libs.utils.search_space as ss
from libs.hparams import get_hparams


def _name(block_name, i, s=None):
    if s is None:
        return '{}.{}'.format(block_name, i)
    return '{}.{}.{}'.format(block_name, i, s)


def _int2str(i, space=ss.CellSpace.CellOps):
    if isinstance(i, str):
        return i
    for k, v in space.items():
        if v == i:
            return k


def _split_op_args(op_code):
    if isinstance(op_code, Sequence) and not isinstance(op_code, str):
        # [op_code, op_arg1, op_arg2, ...]
        op_code, *op_args = op_code
    else:
        op_args = tuple()
    return op_code, op_args


def _get_op_arg(op_args, i, default=None, space=None):
    try:
        result = op_args[i]
        if space is not None:
            index = result
            return space[index]
        else:
            return result
    except IndexError:
        return default


def _get_op_label(op, op_args, hparams, is_combine=False):
    str_op = _int2str(op, space=ss.CellSpace.CombineOps if is_combine else ss.CellSpace.CellOps)
    label_list = [str_op]
    if is_combine:
        if str_op == 'Add':
            pass
        elif str_op == 'Concat':
            pass
        else:
            raise RuntimeError('Unknown op code {}'.format(str_op))
    else:
        if str_op == 'LSTM':
            label_list.extend([
                'reversed={}'.format(_get_op_arg(op_args, 1, False)),
            ])
        elif str_op == 'CNN':
            space = ss.ConvolutionalSpaces[hparams.conv_space]
            label_list.extend([
                'kernel_size={}'.format(_get_op_arg(op_args, 1, 3, space=space.KernelSizes)),
                'stride={}'.format(_get_op_arg(op_args, 2, 1, space=space.Strides)),
                'groups={}'.format(_get_op_arg(op_args, 3, 1, space=space.Groups)),
            ])
        elif str_op == 'SelfAttention':
            space = ss.AttentionSpaces[hparams.attn_space]
            label_list.extend([
                '#heads={}'.format(_get_op_arg(op_args, 0, 8, space=space.NumHeads)),
            ])
        elif str_op == 'FFN':
            space = ss.CellSpace.Activations
            label_list.extend([
                'activ={}'.format(_int2str(_get_op_arg(op_args, 0, space['identity']), space)),
                'bias={}'.format(_get_op_arg(op_args, 1, True)),
            ])
        elif str_op == 'PFFN':
            pass
        elif str_op == 'Identity':
            pass
        elif str_op == 'GroupedLSTM':
            raise NotImplementedError()
        elif str_op == 'EncoderAttention':
            space = ss.AttentionSpaces[hparams.attn_space]
            label_list.extend([
                '#heads={}'.format(_get_op_arg(op_args, 0, 8, space=space.NumHeads)),
            ])
        else:
            raise RuntimeError('Unknown op code {}'.format(str_op))
    return '\n'.join(label_list)


def _make_cell_subgraph(block_name, i, in1, in2, op1_code, op2_code, combine_op_code, hparams):
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

    op1, op1_args = op1_code
    op2, op2_args = op2_code
    combine_op, combine_op_args = combine_op_code

    c.node(i1_n, 'in1', fillcolor='lightblue')
    c.node(i2_n, 'in2', fillcolor='lightblue')
    c.node(op1_n, _get_op_label(op1, op1_args, hparams), fillcolor='green')
    c.node(op2_n, _get_op_label(op2, op2_args, hparams), fillcolor='green')
    c.node(combine_op_n, _get_op_label(combine_op, combine_op_args, hparams, is_combine=True), fillcolor='orange')
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
    parser.add_argument('-H', '--hparams', help='HParams JSON filename, default is use system default', default=None)

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

    if args.hparams is None:
        hparams = get_hparams('normal')
        print('WARNING: Use default hparams, op args may be incorrect. Please give a hparams JSON file.')
    else:
        with open(args.hparams, 'r', encoding='utf-8') as f:
            hparams = argparse.Namespace(**json.load(f))

    g_global = gv.Digraph()

    g_global.format = args.format
    g_global.name = 'G'

    for name, block in blocks.items():
        # [NOTE]: Add 'cluster_' prefix to add border of this subgraph.
        g = gv.Digraph(name='cluster_' + name)
        g.graph_attr.update({
            'label': name,
            'labelloc': 't',
        })
        g.node_attr.update({
            'shape': 'box',
            'style': 'filled',
        })

        # Block-level Combine Node
        bcn_name = _name(name, 'combine')
        g.node(bcn_name, 'combine')

        input_node_indices = []
        for i, cell in enumerate(block, start=0):
            node_name = _name(name, i)
            in1, in2, op1, op2, combine_op = cell

            op1, op_args1 = _split_op_args(op1)
            op2, op_args2 = _split_op_args(op2)
            combine_op, combine_op_args = _split_op_args(combine_op)

            if in1 is None:
                g.node(node_name, 'input {}'.format(len(input_node_indices)))
                input_node_indices.append(i)
            else:
                g.subgraph(_make_cell_subgraph(
                    name, i,
                    in1, in2, (op1, op_args1), (op2, op_args2), (combine_op, combine_op_args), hparams))

                # Add input edges.
                for in_, in_name in zip((in1, in2), ('in1', 'in2')):
                    if in_ in input_node_indices:
                        g.edge(_name(name, in_), _name(name, i, in_name))
                    else:
                        g.edge(_name(name, in_, 'combine'), _name(name, i, in_name))

                g.edge(_name(name, i, 'combine'), bcn_name)

        g_global.subgraph(g)

    print(g_global.source)

    print(g_global.render(filename=output_file, cleanup=True))


if __name__ == '__main__':
    main()
