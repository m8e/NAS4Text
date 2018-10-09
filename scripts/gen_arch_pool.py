#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate a large architecture pool.

Examples
==========

python scripts\gen_arch_pool.py -n 100 -e ignored_scripts\arch_pool.txt -o ignored_scripts\arch_pool.txt \
    --dir-output usr_net_code\arch_pool -H models\de_en_iwslt_bpe2\de_en_iwslt_nao\nao_train_large\hparams.json \
    --cell-op-space only-attn
"""

import argparse
import json
import os
import sys

ProjectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ProjectRoot)

from libs.models.nao_child_net import NAOController
from libs.hparams import get_hparams
from libs.layers.net_code import NetCode

__author__ = 'fyabc'


def main(args=None):
    parser = argparse.ArgumentParser(description='Generate arch pool from net code.')
    parser.add_argument('-H', '--hparams', help='HParams JSON filename, default is use system default', default=None)
    parser.add_argument('--hparams-set', help='HParams set, default is %(default)r', default=None)
    parser.add_argument('--exist', help='Exists arch pool filename', default=None)
    parser.add_argument('-o', '--output', default='ignored_scripts/arch_pool.txt',
                        help = 'Output filename, default is %(default)r')
    parser.add_argument('--dir-output', default='usr_net_code/arch_pool',
                        help='Splitted output directory, default is %(default)r')
    parser.add_argument('--no-dir-output', action='store_true', default=False, help='Disable split output')
    parser.add_argument('-n', type=int, help='Arch pool size, default is %(default)s', default=1000)
    parser.add_argument('--cell-op-space', default=None, help='Specify the cell op space, default is %(default)r')
    parser.add_argument('--global-keys', default='',
                        help='Set comma-separated global keys to search, default is %(default)r')
    parser.add_argument('-e', '--num-encoder-layers', type=int, default=2,
                        help='Number of encoder layers, default is %(default)r')
    parser.add_argument('-d', '--num-decoder-layers', type=int, default=2,
                        help='Number of decoder layers, default is %(default)r')

    args = parser.parse_args(args)

    print(args)

    if args.hparams is None:
        if args.hparams_set is None:
            print('WARNING: Use default hparams, op args may be incorrect. '
                  'Please give a hparams JSON file or specify the hparams set.')
            hparams_set = 'normal'
        else:
            hparams_set = args.hparams_set
        hparams = get_hparams(hparams_set)
    else:
        with open(args.hparams, 'r', encoding='utf-8') as f:
            hparams = argparse.Namespace(**json.load(f))
    if args.cell_op_space is not None:
        hparams.cell_op_space = args.cell_op_space
    hparams.num_encoder_layers = args.num_encoder_layers
    hparams.num_decoder_layers = args.num_decoder_layers

    print('Cell op space:', hparams.cell_op_space)

    controller = NAOController(hparams)

    arch_pool = []

    if args.exist is not None:
        with open(args.exist, 'r', encoding='utf-8') as f:
            for line in f:
                arch_pool.append(NetCode(json.loads(line)))

    _prev_n = len(arch_pool)
    print('Generate: ', end='')
    while len(arch_pool) < args.n:
        new_arch = controller.generate_arch(1, global_keys=args.global_keys.split(','))[0]
        if not controller.valid_arch(new_arch.blocks['enc1'], True) or \
                not controller.valid_arch(new_arch.blocks['dec1'], False):
            continue
        if all(not arch.fast_eq(new_arch) for arch in arch_pool):
            arch_pool.append(new_arch)
        for i in range(_prev_n, len(arch_pool)):
            print((i + 1) if (i + 1) % 100 == 0 else '.', end='')
        _prev_n = len(arch_pool)
    print()

    if not args.no_dir_output:
        split_dir = os.path.join(ProjectRoot, args.dir_output)
        os.makedirs(split_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for i, arch in enumerate(arch_pool, start=1):
            print(json.dumps(arch.original_code), file=f)
            if not args.no_dir_output:
                with open(os.path.join(split_dir, 'arch_{}.json'.format(i)), 'w', encoding='utf-8') as f_split:
                    json.dump(arch.original_code, f_split, indent=4)

    print('Generate {} architectures into {}'.format(args.n, args.output))


if __name__ == '__main__':
    main()
