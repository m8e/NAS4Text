#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate a large architecture pool."""

import argparse
import json
import os
import sys

ProjectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, ProjectRoot)

from libs.models.nao_child_net import NAOController
from libs.hparams import get_hparams

__author__ = 'fyabc'


def main(args=None):
    parser = argparse.ArgumentParser(description='Generate arch pool from net code.')
    parser.add_argument('-H', '--hparams', help='HParams JSON filename, default is use system default', default=None)
    parser.add_argument('-o', '--output', help='Output filename', default=None)
    parser.add_argument('--dir-output', help='Splitted output directory', default='usr_net_code/arch_pool')
    parser.add_argument('-n', type=int, help='Arch pool size, default is %(default)s', default=1000)

    args = parser.parse_args(args)

    if args.hparams is None:
        hparams = get_hparams('normal')
        print('WARNING: Use default hparams, op args may be incorrect. Please give a hparams JSON file.')
    else:
        with open(args.hparams, 'r', encoding='utf-8') as f:
            hparams = argparse.Namespace(**json.load(f))

    controller = NAOController(hparams)

    arch_pool = []
    _prev_n = 0
    print('Generate: ', end='')
    while len(arch_pool) < args.n:
        new_arch = controller.generate_arch(1)[0]
        if not controller.valid_arch(new_arch.blocks['enc1'], True) or \
                not controller.valid_arch(new_arch.blocks['dec1'], False):
            continue
        if all(not arch.fast_eq(new_arch) for arch in arch_pool):
            arch_pool.append(new_arch)
        for i in range(_prev_n, len(arch_pool)):
            print((i + 1) if (i + 1) % 100 == 0 else '.', end='')
        _prev_n = len(arch_pool)
    print()

    split_dir = os.path.join(ProjectRoot, args.dir_output)
    os.makedirs(split_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        for i, arch in enumerate(arch_pool, start=1):
            print(json.dumps(arch.original_code), file=f)
            with open(os.path.join(split_dir, 'arch_{}.json'.format(i)), 'w', encoding='utf-8') as f_split:
                json.dump(arch.original_code, f_split, indent=4)

    print('Generate {} architectures into {}'.format(args.n, args.output))


if __name__ == '__main__':
    main()
