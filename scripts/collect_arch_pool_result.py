#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Collect training result of arch pool."""

import argparse
import os

import numpy as np

__author__ = 'fyabc'

PathTemplate = 'F:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/log-iter{}/'
FnTemplate = 'de_en_iwslt_bpe2-{}-arch_{}-base-generate-{}.log.txt'
TargetFileTemplates = {
    'x': 'F:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/arches-{}{}.txt',
    'y': 'F:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/bleus-{}{}.txt',
}
PoolFileTemplate = 'F:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/arch_pool-{}{}.txt'

Baseline = {
    'dev': {
        2: 33.72,
        4: 34.86,
        6: 34.96,
    },
    'test': {
        2: 32.74,
        4: 33.71,
        6: 33.74,
    },
}


def _find_max(l, k=5):
    l = np.array([-100 if e is None else e for e in l])
    topk_index = np.argsort(-l)[:k]
    return l[topk_index].tolist(), (topk_index + 1).tolist()


def _find_min(l):
    index = None
    min_val = 100
    for i, b in enumerate(l):
        if b is None:
            continue
        if b < min_val:
            index = i
            min_val = b
    return min_val, index + 1


def compare_baseline(bleu_list, subset, num_layers=2):
    baseline = Baseline[subset][num_layers]
    num_higher = sum(int(b is not None and b > baseline) for b in bleu_list)
    return num_higher


def real_main(hparams):
    not_exist, empty = 0, []
    n_arches = hparams.end + 1 - hparams.start
    iteration = hparams.iteration
    subset = hparams.subset
    hparams_set = hparams.hparams_set

    bleu_list = [None for _ in range(hparams.start, hparams.end + 1)]
    for i in range(hparams.start, hparams.end + 1):
        print(i, end=' ')
        fname = os.path.join(PathTemplate.format(hparams.iteration), FnTemplate.format(hparams_set, i, subset))

        if not os.path.exists(fname):
            print('Not exist')
            not_exist += 1
            continue

        with open(fname, 'r', encoding='utf-8') as f:
            lines = list(f)

            if not lines:
                print('Empty')
                empty.append(i)
            else:
                last_line = lines[-1]
                bleu = float(last_line.split(' ')[2].strip(','))
                bleu_list[i - hparams.start] = bleu
                print(bleu)

    n_empty = len(empty)
    print(''.center(40, '='))
    print('Exist:', n_arches - not_exist - n_empty, 'Not exist:', not_exist, 'Empty:', n_empty)
    print('Empty:', *empty)
    print('Max: {} at {}'.format(*_find_max(bleu_list)), 'Min: {} at {}'.format(*_find_min(bleu_list)))

    num_layers = 2
    print('{} higher than baseline {} on {} set'.format(
        compare_baseline(bleu_list, subset, num_layers), num_layers, subset))

    line_list_out = []
    bleu_list_out = []
    with open(PoolFileTemplate.format(hparams_set, iteration), 'r', encoding='utf-8') as f_pool:
        print('Read architectures from {!r}'.format(f_pool.name))
        lines = [l.strip() for l in f_pool]
        assert len(lines) == len(bleu_list), 'Arch pool size {} != arch BLEU size {}'.format(len(lines), len(bleu_list))
        for i, bleu in enumerate(bleu_list):
            if bleu is not None:
                line_list_out.append(lines[i])
                bleu_list_out.append(bleu)
    with open(TargetFileTemplates['x'].format(hparams_set, iteration), 'w', encoding='utf-8') as f_x, \
            open(TargetFileTemplates['y'].format(hparams_set, iteration), 'w', encoding='utf-8') as f_y:
        for line, bleu in zip(line_list_out, bleu_list_out):
            print(line, file=f_x)
            print(bleu, file=f_y)
        print('Dump results into {!r} and {!r}.'.format(f_x.name, f_y.name))


def main(args=None):
    parser = argparse.ArgumentParser(description='Collect architecture pool result.')
    parser.add_argument('-i', '--iteration', type=int, default=1, help='The iteration number, default is %(default)r')
    parser.add_argument('-s', '--start', type=int, default=1, help='The start arch id, default is %(default)r')
    parser.add_argument('-e', '--end', type=int, default=1000, help='The end arch id, default is %(default)r')
    parser.add_argument('--subset', default='dev', help='The subset, default is %(default)r')
    parser.add_argument('-H', '--hparams-set', default='de_en_iwslt_nao',
                        help='The hparams, set, default is %(default)s')

    hparams = parser.parse_args(args)
    print(hparams)

    real_main(hparams)


if __name__ == '__main__':
    main('-i 2 -s 1001 -e 1500 --subset dev'.split(' '))
