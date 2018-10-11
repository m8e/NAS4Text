#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'fyabc'

ColorList = ['m', 'k', 'g', 'c', 'y', 'r', 'b', 'chartreuse', 'peru']
StyleList = ['-', '--', '.', '-.', ':']


def extract_dev_ppl(filename):
    epoch_list = []
    ppl_list = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split()
            if 'valid_ppl' not in words:
                continue
            epoch_list.append(int(words[2].lstrip('0')))
            ppl_list.append(float(words[16]))

    return epoch_list, ppl_list


def extract_train_ppl(filename):
    epoch_list = []
    ppl_list = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split()
            if 'nll_loss' not in words:
                continue
            epoch_list.append(int(words[2].lstrip('0')))
            ppl_list.append(float(words[11]))

    return epoch_list, ppl_list


def init():
    font = {
        # 'weight': 'bold',
        'size': 20,
    }
    lines = {
        'linewidth': 3,
    }
    matplotlib.rc('font', **font)
    matplotlib.rc('lines', **lines)


def get_baseline(ed_list, extract_fn):
    return [
        ('Baseline-e{}d{}'.format(e, d),
         extract_fn('D:/DataTransfer/NAS4Text/logs/train-e{}d{}_baseline.txt'.format(e, d)))
        for e, d in ed_list
    ]


def get_fairseq_baseline(eds_list, extract_fn):
    return [
        ('Baseline-fairseq-e{}d{}-{}'.format(e, d, s),
         extract_fn('D:/DataTransfer/NAS4Text/logs/train-fairseq-e{}d{}_baseline-seed{}.txt'.format(e, d, s)))
        for e, d, s in eds_list
    ]


def get_attn(ed_list, extract_fn, attn):
    return [
        ('Attn{}-e{}d{}'.format(attn, e, d),
         extract_fn('D:/DataTransfer/NAS4Text/logs/train-e{}d{}_baseline_attn{}.txt'.format(e, d, attn)))
        for e, d in ed_list
    ]


def get_norm_before(ed_list, extract_fn):
    return [
        ('NormBefore-e{}d{}'.format(e, d),
         extract_fn('D:/DataTransfer/NAS4Text/logs/train-e{}d{}_baseline_n_da.txt'.format(e, d)))
        for e, d in ed_list
    ]


def baseline_style(key):
    if key.startswith('Baseline'):
        return '-'
    if key.startswith('NormBefore'):
        return '-.'
    return '--'


Styles = {
    'baseline': baseline_style,
}


def get_style(style_name, key):
    return Styles[style_name](key)


def plot_ppl(args: dict):
    subset = args['subset']
    extract_fn = extract_dev_ppl if subset == 'dev' else extract_train_ppl

    e_ppl_dict = OrderedDict(
        get_baseline([
            (2, 2),
            (4, 4),
            (6, 6),
            # (4, 8),
            # (8, 4),
            (8, 8),
            (10, 10),
            # (12, 12),
        ], extract_fn) + get_fairseq_baseline([
            # (2, 2, 1),
            # (2, 2, 2),
            # (2, 2, 3),
            # (4, 4, 1),
            # (4, 4, 2),
            # (6, 6, 1),
            # (6, 6, 2),
            # (6, 6, 3),
            # (4, 8, 1),
            # (8, 4, 1),
            # (8, 8, 1),
            # (10, 10, 1),
            # (12, 12, 1),
        ], extract_fn) + get_attn([
            (6, 6),
            (8, 8),
            (10, 10),
        ], extract_fn, 384) + get_norm_before([
            (8, 8),
            (10, 10),
        ], extract_fn) + [
            # ('1141', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141.txt')),
            # ('1141-e4d4', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4.txt')),
            # ('1141-e4d4-common-dp+=0.1', extract_fn(
            #     'D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_common_dp_add_01.txt')),
            # ('1141-e4d4-attn-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_attn_dp_add_01.txt')),
            # ('1141-e4d4-ffn-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_ffn_dp_add_01.txt')),
            # ('1141-e4d4-ppp-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_ppp_dp_add_01.txt')),
            # ('1295', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295.txt')),
            # ('1295-e4d4', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4.txt')),
            # ('1295-e4d4-common-dp+=0.1', extract_fn(
            #     'D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_common_dp_add_01.txt')),
            # ('1295-e4d4-attn-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_attn_dp_add_01.txt')),
            # ('1295-e4d4-ffn-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_ffn_dp_add_01.txt')),
            # ('1295-e4d4-ppp-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_ppp_dp_add_01.txt')),
        ])

    for i, (key, (e_list, ppl_list)) in enumerate(e_ppl_dict.items()):
        plt.plot(e_list, ppl_list, get_style('baseline', key), color=ColorList[i % len(ColorList)], label=key)

    plt.xticks(np.arange(0, 61, 5))
    if subset == 'dev':
        # plt.yticks(np.arange(6.00, 20.00, 2.00))
        # plt.ylim(ymin=6.00, ymax=20.00)
        pass
    else:
        # plt.yticks(np.arange(3.00, 20.00, 2.00))
        # plt.ylim(ymin=3.00, ymax=20.00)
        pass
    # t = np.arange(5.00, 25.00, 0.01)
    # plt.semilogy(t, np.exp(-t / 1.0))
    plt.xlim(xmin=0, xmax=61)

    # plt.title('Test Accuracy')
    plt.title('{} PPL'.format(subset))
    plt.grid(which='both')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()


def main():
    init()
    plot_ppl({
        'subset': 'dev',
    })


if __name__ == '__main__':
    main()
