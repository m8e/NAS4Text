#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Collect training result of arch pool."""

import argparse
import os
import random

import numpy as np

__author__ = 'fyabc'

PathTemplate = 'D:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/log-iter{}/'
FnTemplate = 'de_en_iwslt_bpe2-{}-{}_{}-base-generate-{}.log.txt'
TargetFileTemplates = {
    'x': 'D:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/arches-{}-{}-{}.txt',
    'y': 'D:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/bleus-{}-{}-{}.txt',
}
PoolFileTemplate = 'D:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results/arch_pool-{}{}.txt'

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

# Focus top, must in increase order of rep and decrease order of rank.
FocusTop = [[200, 1], [100, 2], [50, 3], [10, 5]]
# Focus bottom, must in increase order of rank.
FocusBottom = [[150, 0.5]]


def _find_max(l, k=5, start=1):
    l = np.array([-100 if e is None else e for e in l])
    topk_index = np.argsort(-l)[:k]
    return l[topk_index].tolist(), (topk_index + start).tolist()


def _find_min(l, k=5, start=1):
    l = np.array([100 if e is None else e for e in l])
    topk_index = np.argsort(l)[:k]
    return l[topk_index].tolist(), (topk_index + start).tolist()


def _average(l):
    s, n = 0.0, 0
    for b in l:
        if b is None:
            continue
        s += b
        n += 1
    return s / n


def compare_baseline(bleu_list, subset, num_layers=2):
    baseline = Baseline[subset][num_layers]
    num_higher = sum(int(b is not None and b > baseline) for b in bleu_list)
    return num_higher


def focus_top(bleu_list: list) -> list:
    """Get the index list that focus top arches and random drop bottom arches.

    Replicate top arches, random drop bottom arches.
    top10 *= 4, top50 *= 3, top100 *= 2, top200 *= 1, bottom150 *= 0.5 (rand-drop 0.5).

    Args:
        bleu_list (list):

    Returns:
        list: index list.
    """
    bleu_list = np.array(bleu_list)
    argsort_index = np.argsort(bleu_list)

    result = argsort_index.tolist()     # type: list

    current_rank = 0
    to_be_replaced = []
    for rank, p in FocusBottom:
        to_be_replaced.extend(random.sample(result[current_rank:rank], int(p * (rank - current_rank))))
        current_rank = rank
    result[:current_rank] = to_be_replaced

    current_rep = 0
    for rank, rep in FocusTop:
        result.extend(argsort_index[-rank:].tolist() * (rep - current_rep))
        current_rep = rep

    return result


def real_main(hparams):
    not_exist, empty = 0, []
    n_arches = hparams.end + 1 - hparams.start
    iteration = hparams.iteration
    subset = hparams.subset
    hparams_set = hparams.hparams_set

    bleu_list = [None for _ in range(hparams.start, hparams.end + 1)]
    for i in range(hparams.start, hparams.end + 1):
        print(i, end=' ')
        fname = os.path.join(PathTemplate.format(hparams.iteration), FnTemplate.format(
            hparams_set, hparams.arch_pool, i, subset))

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
    print('Max: {} at {}'.format(*_find_max(bleu_list, k=10, start=hparams.start)))
    print('Min: {} at {}'.format(*_find_min(bleu_list, k=3, start=hparams.start)))
    print('Average: {}'.format(_average(bleu_list)))

    num_layers = hparams.num_layers
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

    # Read old results.
    line_list_final, bleu_list_final = [], []
    for i in range(1, iteration):
        with open(TargetFileTemplates['x'].format(hparams_set, subset, i), 'r', encoding='utf-8') as f_x2:
            for line in f_x2:
                line_list_final.append(line.strip())
        with open(TargetFileTemplates['y'].format(hparams_set, subset, i), 'r', encoding='utf-8') as f_y2:
            for line in f_y2:
                bleu_list_final.append(float(line))
        print('Concat result of iteration {}'.format(i))
    line_list_final.extend(line_list_out)
    bleu_list_final.extend(bleu_list_out)
    assert len(line_list_final) == len(bleu_list_final)

    # Replicate top arches, random drop bottom arches.
    if hparams.focus:
        index_final = focus_top(bleu_list_final)
        line_list_final = [line_list_final[i] for i in index_final]
        bleu_list_final = [bleu_list_final[i] for i in index_final]

    with open(TargetFileTemplates['x'].format(hparams_set, subset, iteration), 'w', encoding='utf-8') as f_x, \
            open(TargetFileTemplates['y'].format(hparams_set, subset, iteration), 'w', encoding='utf-8') as f_y:
        for line, bleu in zip(line_list_final, bleu_list_final):
            print(line, file=f_x)
            print(bleu, file=f_y)
        print('Dump {} results into {!r} and {!r}.'.format(len(line_list_final), f_x.name, f_y.name))


def main(args=None):
    parser = argparse.ArgumentParser(description='Collect architecture pool result.')
    parser.add_argument('-i', '--iteration', type=int, default=1, help='The iteration number, default is %(default)r')
    parser.add_argument('-s', '--start', type=int, default=1, help='The start arch id, default is %(default)r')
    parser.add_argument('-e', '--end', type=int, default=1000, help='The end arch id, default is %(default)r')
    parser.add_argument('--subset', default='dev', help='The subset, default is %(default)r')
    parser.add_argument('-H', '--hparams-set', default='de_en_iwslt_nao',
                        help='The hparams, set, default is %(default)s')
    parser.add_argument('-a', '--arch-pool', default='arch', help='Arch pool name, default is %(default)r')
    parser.add_argument('-n', '--num-layers', default=2, type=int, help='Number of layers, default is %(default)r')
    parser.add_argument('-f', '--focus', action='store_true', default=False, help='Focus on top arches')

    hparams = parser.parse_args(args)
    print(hparams)

    real_main(hparams)


if __name__ == '__main__':
    # main('-i 1 -s 1 -e 1000 --subset dev -a arch_pool_e6d6_dp -n 6'.split(' '))

    # main('-i 2 -s 1001 -e 1500 --subset dev -a arch_pool_e6d6_dp -n 6'.split(' '))
    # main('-i 2 -s 1001 -e 1500 --subset test -a arch_pool_e6d6_dp -n 6'.split(' '))
    # main('-i 2 -s 1001 -e 1500 --subset dev'.split(' '))

    # main('-i 3 -s 1501 -e 2000 --subset dev -a arch_pool_e6d6_dp -n 6'.split(' '))
    main('-i 3 -s 1501 -e 2000 --subset test -a arch_pool_e6d6_dp -n 6'.split(' '))
    # main('-i 3 -s 1501 -e 1800 --subset dev'.split(' '))
    pass
