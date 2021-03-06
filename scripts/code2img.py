#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate image from net code.

Small mapping:
    block == layer
    cell == node
    op == branch
"""

import argparse
from collections import Sequence, defaultdict
import json
import os
import sys

import graphviz as gv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.layers.net_code import load_net_code_from_file, NetCode
import libs.utils.search_space as ss
from libs.hparams import get_hparams


def _name(layer_name, i, s=None):
    if s is None:
        return '{}.{}'.format(layer_name, i)
    return '{}.{}.{}'.format(layer_name, i, s)


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


def _get_op_label(op, op_args, hparams, is_combine=False, node_args=()):
    str_op = _int2str(op, space=ss.CellSpace.CombineOps if is_combine else ss.CellSpace.CellOps)
    label_list = [str_op]
    pre_list, post_list = [], []
    if is_combine:
        # [NOTE]: Maybe postprocessors on combine
        post_list = ss.PPPSpace.get_ops(_get_op_arg(node_args, 1, ''))
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
            pre_list = ss.PPPSpace.get_ops(_get_op_arg(op_args, 1, ''))
            post_list = ss.PPPSpace.get_ops(_get_op_arg(op_args, 2, ''))
        elif str_op == 'FFN':
            space = ss.CellSpace.Activations
            label_list.extend([
                'activ={}'.format(_int2str(_get_op_arg(op_args, 0, space['identity']), space)),
                'bias={}'.format(_get_op_arg(op_args, 1, True)),
            ])
        elif str_op == 'PFFN':
            pre_list = ss.PPPSpace.get_ops(_get_op_arg(op_args, 0, ''))
            post_list = ss.PPPSpace.get_ops(_get_op_arg(op_args, 1, ''))
            space = ss.AttentionSpaces[hparams.attn_space]
            label_list.extend([
                '#dim={}'.format(_get_op_arg(op_args, 2, default=hparams.attn_d_ff, space=space.FFNSize)),
            ])
        elif str_op == 'Identity':
            pass
        elif str_op == 'Zero':
            pass
        elif str_op == 'GroupedLSTM':
            raise NotImplementedError()
        elif str_op == 'EncoderAttention':
            space = ss.AttentionSpaces[hparams.attn_space]
            label_list.extend([
                '#heads={}'.format(_get_op_arg(op_args, 0, 8, space=space.NumHeads)),
            ])
            pre_list = ss.PPPSpace.get_ops(_get_op_arg(op_args, 1, ''))
            post_list = ss.PPPSpace.get_ops(_get_op_arg(op_args, 2, ''))
        else:
            raise RuntimeError('Unknown op code {}'.format(str_op))
    total_list = pre_list + ['\\n'.join(label_list)] + post_list
    return '{{{}}}'.format('|'.join(total_list))


def _make_node_subgraph(layer_name, i, in1, in2, op1_code, op2_code, combine_op_code, hparams, node_args=()):
    c = gv.Digraph(name='cluster_{}_{}'.format(layer_name, i))
    c.graph_attr.update({
        'label': 'Node {}'.format(i),
        'labelloc': 't',
    })
    c.node_attr.update({
        'shape': 'record',
        'style': 'filled',
    })

    i1_n = _name(layer_name, i, 'in1')
    i2_n = _name(layer_name, i, 'in2')
    op1_n = _name(layer_name, i, 'op1')
    op2_n = _name(layer_name, i, 'op2')
    combine_op_n = _name(layer_name, i, 'combine')

    op1, op1_args = op1_code
    op2, op2_args = op2_code
    combine_op, combine_op_args = combine_op_code

    # [NOTE]: Maybe preprocessors on in1 and in2
    preprocessors = ss.PPPSpace.get_ops(_get_op_arg(node_args, 0, ''))

    c.node(i1_n, '{{{}}}'.format('|'.join(preprocessors + ['in1'])), fillcolor='lightblue')
    c.node(i2_n, '{{{}}}'.format('|'.join(preprocessors + ['in2'])), fillcolor='lightblue')
    c.node(op1_n, _get_op_label(op1, op1_args, hparams), fillcolor='green')
    c.node(op2_n, _get_op_label(op2, op2_args, hparams), fillcolor='green')
    c.node(combine_op_n,
           _get_op_label(combine_op, combine_op_args, hparams, is_combine=True, node_args=node_args),
           fillcolor='orange')
    c.edge(i1_n, op1_n)
    c.edge(i2_n, op2_n)
    c.edge(op1_n, combine_op_n)
    c.edge(op2_n, combine_op_n)

    return c


def _replace_default_layer_name(name):
    if name == 'enc1':
        return 'Encoder Layer'
    if name == 'dec1':
        return 'Decoder Layer'
    return name


def main(args=None):
    parser = argparse.ArgumentParser(description='Generate image from net code.')
    parser.add_argument('file', help='Net code file')
    parser.add_argument('-T', '--format', default='jpg', help='Output format, default is %(default)r')
    parser.add_argument('-d', '--dir', help='Output directory', required=True)
    parser.add_argument('-o', '--output', help='Output filename (without format ext)', default=None)
    parser.add_argument('-H', '--hparams', help='HParams JSON filename, default is use system default', default=None)
    parser.add_argument('-s', '--show', action='store_true', default=False,
                        help='Show the output image, default is %(default)r')

    parser.add_argument('--nt', '--no-title', dest='title', action='store_false', default=True,
                        help='Does not show the title')
    parser.add_argument('--fs', '--fontsize', dest='fontsize', default=None,
                        help='Set the global font size, default is %(default)r')

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
        layers = code.blocks
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

    for kw in 'graph', 'node', 'edge':
        g_global.attr(kw, fontname='Times-bold')
    if args.fontsize is not None:
        # Default font size of Graphviz is 14.
        for kw in 'graph', 'node', 'edge':
            g_global.attr(kw, fontsize=args.fontsize)
    g_global.attr('edge', color='blue', penwidth='7')

    layer_counter = defaultdict(int)
    for layers_ed in code.original_code['Layers']:  # Get layer in string format
        for layer in layers_ed:
            if isinstance(layer, str):
                layer_counter[layer] += 1
            else:
                layer_counter['<unknown>'] += 1

    if args.title:
        global_title_list = [
            'Name={}'.format(output_basename),
            ', '.join('{}*{}'.format(b, n) for b, n in layer_counter.items()),
            ', '.join('{}={}'.format(key, getattr(ss.GlobalSpace, key)[index]) for key, index in code.global_code.items())
        ]
        global_title = '\n'.join(global_title_list)
        g_global.graph_attr.update({
            'label': global_title,
            'labelloc': 't',
        })

    for name, layer in layers.items():
        # [NOTE]: Add 'cluster_' prefix to add border of this subgraph.
        g = gv.Digraph(name='cluster_' + name)
        g.graph_attr.update({
            'label': _replace_default_layer_name(name),
            'labelloc': 't',
        })
        g.node_attr.update({
            'shape': 'box',
            'style': 'filled',
        })

        # layer-level Combine Node
        lcn_name = _name(name, 'combine')
        g.node(lcn_name, hparams.block_combine_op)

        # if name == 'enc1':
        # g.body.append('newrank=true')

        input_node_indices = []
        all_nodes = set()
        used_nodes = set()
        for i, node in enumerate(layer, start=0):
            if isinstance(node, dict):
                # TODO: Process layer params.
                continue

            node_name = _name(name, i)
            in1, in2, op1, op2, combine_op, *node_args = node

            op1, op_args1 = _split_op_args(op1)
            op2, op_args2 = _split_op_args(op2)
            combine_op, combine_op_args = _split_op_args(combine_op)

            if in1 is None:
                g.node(node_name, 'Input {}'.format(len(input_node_indices)))
                input_node_indices.append(i)
            else:
                all_nodes.add(i)
                used_nodes.update({in1, in2})

                g.subgraph(_make_node_subgraph(
                    name, i,
                    in1, in2, (op1, op_args1), (op2, op_args2), (combine_op, combine_op_args), hparams, node_args))

                # Add input edges.
                for in_, in_name in zip((in1, in2), ('in1', 'in2')):
                    if in_ in input_node_indices:
                        g.edge(_name(name, in_), _name(name, i, in_name))
                    else:
                        g.edge(_name(name, in_, 'combine'), _name(name, i, in_name))

        if hparams.block_combine_no_outs:
            out_nodes = all_nodes - used_nodes
        else:
            out_nodes = all_nodes
        for i in out_nodes:
            g.edge(_name(name, i, 'combine'), lcn_name)

        # Force all input nodes to at the same rank.
        g.body.append('{{rank=same; {}}}'.format('; '.join('"{}"'.format(_name(name, i)) for i in input_node_indices)))

        g_global.subgraph(g)

    print(g_global.source)

    print(g_global.render(filename=output_file, cleanup=True, view=args.show))


if __name__ == '__main__':
    main()
