#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Collect training result of arch pool."""

import os

import numpy as np

__author__ = 'fyabc'

Subset = 'dev'
HParams = 'de_en_iwslt_nao'
Path = 'F:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/log/'
FnTemplate = 'de_en_iwslt_bpe2-{}-arch_{{}}-base-generate-{}.log.txt'.format(HParams, Subset)
TargetFiles = {
    'x': 'F:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/arches-{}.txt'.format(HParams),
    'y': 'F:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/bleus-{}.txt'.format(HParams),
}
PoolFile = 'F:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/arch_pool-{}.txt'.format(HParams)
N = 1000

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


def compare_baseline(bleu_list, num_layers=2):
    baseline = Baseline[Subset][num_layers]
    num_higher = sum(int(b is not None and b > baseline) for b in bleu_list)
    return num_higher


def main():
    not_exist, empty = 0, []
    bleu_list = [None for _ in range(N)]
    for i in range(1, N + 1):
        print(i, end=' ')
        fname = os.path.join(Path, FnTemplate.format(i))
    
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
                bleu_list[i - 1] = bleu
                print(bleu)

    n_empty = len(empty)
    print(''.center(40, '='))
    print('Exist:', N - not_exist - n_empty, 'Not exist:', not_exist, 'Empty:', n_empty)
    print('Empty:', *empty)
    print('Max: {} at {}'.format(*_find_max(bleu_list)), 'Min: {} at {}'.format(*_find_min(bleu_list)))

    num_layers = 2
    print('{} higher than baseline {} on {} set'.format(compare_baseline(bleu_list, num_layers), num_layers, Subset))

    line_list_out = []
    bleu_list_out = []
    with open(PoolFile, 'r', encoding='utf-8') as f_pool:
        lines = [l.strip() for l in f_pool]
        for i, bleu in enumerate(bleu_list):
            if bleu is not None:
                line_list_out.append(lines[i])
                bleu_list_out.append(bleu)
    with open(TargetFiles['x'], 'w', encoding='utf-8') as f_x, \
            open(TargetFiles['y'], 'w', encoding='utf-8') as f_y:
        for line, bleu in zip(line_list_out, bleu_list_out):
            print(line, file=f_x)
            print(bleu, file=f_y)
    print('Dump results into {!r} and {!r}.'.format(TargetFiles['x'], TargetFiles['y']))


if __name__ == '__main__':
    main()
