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
        'markersize': 10,
    }
    matplotlib.rc('font', **font)
    matplotlib.rc('lines', **lines)


def get_key_name(extract_fn, e, d, s, ffn=None, n_da=False, dp=None, fairseq=False):
    fs = 'fairseq-' if fairseq else ''
    is_baseline = 'Baseline'
    head = ''
    if ffn is not None:
        is_baseline = ''
        head += 'FFN{}-'.format(ffn)
    if n_da:
        is_baseline = ''
        head += 'NormBefore-'
    if dp is not None:
        is_baseline = ''
        head += 'Dropout{}-'.format(dp)

    head = is_baseline + head + '{}e{}d{}-{}'.format(fs, e, d, s)
    return head, extract_fn(
        'D:/DataTransfer/NAS4Text/logs/train-{fairseq}e{e}d{d}_baseline{ffn}{n_da}{dp}-seed{s}.txt'.format(
            fairseq=fs, e=e, d=d, s=s,
            ffn='_ffn{}'.format(ffn) if ffn is not None else '',
            n_da='_n_da' if n_da else '',
            dp='_dp{}'.format(str(dp).replace('.', '')) if dp is not None else '',
        ))


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


def _plot(e_ppl_dict, subset, **kwargs):
    style = kwargs.pop('style', None)
    if style is None:
        def style_fn(key):
            return get_style('baseline', key)
    else:
        style_fn = style

    for i, (key, (e_list, ppl_list)) in enumerate(e_ppl_dict.items()):
        plt.plot(e_list, ppl_list, style_fn(key), color=ColorList[i % len(ColorList)], label=key)

    x_range = kwargs.pop('x_range', np.arange(0, 61, 5))

    plt.xticks(x_range)
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
    plt.xlim(xmin=np.min(x_range), xmax=np.max(x_range))

    # plt.title('Test Accuracy')
    title = '{} PPL'.format(subset)
    extra_title = kwargs.pop('extra_title', None)
    if extra_title:
        title += '\n' + extra_title
    plt.title(title)

    plt.grid(which='both')
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()


def plot_deen_ppl(args: dict):
    subset = args['subset']
    extract_fn = extract_dev_ppl if subset == 'dev' else extract_train_ppl

    e_ppl_dict = OrderedDict([
        # get_key_name(extract_fn, 2, 2, 1),
        # get_key_name(extract_fn, 4, 4, 1),
        # get_key_name(extract_fn, 6, 6, 1),
        # get_key_name(extract_fn, 4, 8, 1),
        # get_key_name(extract_fn, 8, 4, 1),
        # get_key_name(extract_fn, 8, 8, 1),
        # get_key_name(extract_fn, 10, 10, 1),
        # get_key_name(extract_fn, 12, 12, 1),
        #
        # get_key_name(extract_fn, 6, 6, 1, ffn=384, fairseq=False),
        # get_key_name(extract_fn, 6, 6, 1, ffn=1024, fairseq=False),
        # get_key_name(extract_fn, 8, 8, 1, ffn=384, fairseq=False),
        # get_key_name(extract_fn, 10, 10, 1, ffn=384, fairseq=False),
        #
        # get_key_name(extract_fn, 8, 8, 1, n_da=True, fairseq=False),
        # get_key_name(extract_fn, 10, 10, 1, n_da=True, fairseq=False),

        # get_key_name(extract_fn, 2, 2, 1, fairseq=True),
        # get_key_name(extract_fn, 2, 2, 2, fairseq=True),
        # get_key_name(extract_fn, 2, 2, 3, fairseq=True),
        # get_key_name(extract_fn, 4, 4, 1, fairseq=True),
        # get_key_name(extract_fn, 4, 4, 2, fairseq=True),
        # get_key_name(extract_fn, 6, 6, 1, fairseq=True),
        # get_key_name(extract_fn, 6, 6, 2, fairseq=True),
        # get_key_name(extract_fn, 6, 6, 3, fairseq=True),
        # get_key_name(extract_fn, 4, 8, 1, fairseq=True),
        # get_key_name(extract_fn, 8, 4, 1, fairseq=True),
        get_key_name(extract_fn, 8, 8, 1, fairseq=True),
        get_key_name(extract_fn, 10, 10, 1, fairseq=True),
        # get_key_name(extract_fn, 12, 12, 1, fairseq=True),

        # get_key_name(extract_fn, 6, 6, 1, ffn=384, fairseq=True),
        # get_key_name(extract_fn, 6, 6, 1, ffn=1024, fairseq=True),
        # get_key_name(extract_fn, 8, 8, 1, ffn=384, fairseq=True),
        # get_key_name(extract_fn, 10, 10, 1, ffn=384, fairseq=True),
        #
        get_key_name(extract_fn, 8, 8, 1, n_da=True, fairseq=True),
        get_key_name(extract_fn, 10, 10, 1, n_da=True, fairseq=True),

        # get_key_name(extract_fn, 8, 8, 1, dp=0.1, fairseq=True),
        # get_key_name(extract_fn, 8, 8, 1, dp=0.3, fairseq=True),
        # get_key_name(extract_fn, 10, 10, 1, dp=0.1, fairseq=True),

        get_key_name(extract_fn, 8, 8, 1, dp=0.3, n_da=True, fairseq=True),
        get_key_name(extract_fn, 10, 10, 1, dp=0.3, n_da=True, fairseq=True),
        # get_key_name(extract_fn, 12, 12, 1, dp=0.3, n_da=True, fairseq=True),

        # ('1141', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141.txt')),
        # ('1141-e4d4', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4.txt')),
        # ('1141-e4d4-NormBefore', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_n_da.txt')),
        # ('1141-e4d4-common-dp0.1', extract_fn(
        #     'D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_common_dp_add_01.txt')),
        # ('1141-e4d4-attn-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_attn_dp_add_01.txt')),
        # ('1141-e4d4-ffn-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_ffn_dp_add_01.txt')),
        # ('1141-e4d4-ppp-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1141_e4d4_ppp_dp_add_01.txt')),
        # ('1295', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295.txt')),
        # ('1295-e4d4', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4.txt')),
        # ('1295-e4d4-NormBefore', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_n_da.txt')),
        # ('1295-e4d4-common-dp+=0.1', extract_fn(
        #     'D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_common_dp_add_01.txt')),
        # ('1295-e4d4-attn-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_attn_dp_add_01.txt')),
        # ('1295-e4d4-ffn-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_ffn_dp_add_01.txt')),
        # ('1295-e4d4-ppp-dp+=0.1', extract_fn('D:/DataTransfer/NAS4Text/logs/train-arch_1295_e4d4_ppp_dp_add_01.txt')),
    ])

    _plot(e_ppl_dict, subset)


def _simple_style_fn(key):
    if 'baseline' in key:
        return '-'
    if '1886' in key:
        return '--'
    if '1963' in key:
        return '-.'
    return '-'


def plot_ende_ppl(args: dict):
    subset = args['subset']
    extract_fn = extract_dev_ppl if subset == 'dev' else extract_train_ppl

    e_ppl_dict = OrderedDict([
        ('fairseq-baseline', extract_fn('../log/ende-logs/fairseq-vaswani_ende_e6d6_baseline-train.log.txt')),
        ('baseline-time', extract_fn('../log/ende-logs/vaswani_ende_e6d6_baseline-u32-train.log.txt')),
        ('baseline', extract_fn('../log/ende-logs/vaswani_ende_e6d6_baseline-u32-bf-train.log.txt')),
        ('baseline-u16', extract_fn('../log/ende-logs/vaswani_ende_e6d6_baseline-bf-train.log.txt')),

        ('1886', extract_fn('../log/ende-logs/ende_1886-u32-train.log.txt')),
        ('1886-dp05', extract_fn('../log/ende-logs/ende_1886-dp05-u32-train.log.txt')),
        ('1963', extract_fn('../log/ende-logs/ende_1963-u32-train.log.txt')),
        ('1963-dp05', extract_fn('../log/ende-logs/ende_1963-dp05-u32-train.log.txt')),
    ])

    _plot(
        e_ppl_dict, subset,
        x_range=np.arange(0, 41, 5),
        extra_title='Default: batch_first, update_freq=32, dropout=0.3',
        style=_simple_style_fn,
    )


def plot_deen_ppl2(args: dict):
    subset = args['subset']
    extract_fn = extract_dev_ppl if subset == 'dev' else extract_train_ppl

    e_ppl_dict = OrderedDict([
        ('fairseq-baseline', extract_fn('../log/deen-logs/train-fairseq-e6d6_baseline-seed1.txt')),
        # ('baseline', extract_fn('../log/deen-logs/e6d6_baseline-train.log.txt')),
        # ('baseline-linux', extract_fn('../log/deen-logs/e6d6_baseline-train-linux.log.txt')),
        # ('baseline-time_first', extract_fn('../log/deen-logs/e6d6_baseline-train-time_first.log.txt')),
        ('baseline-time_first-new', extract_fn('../log/deen-logs/e6d6_baseline-train-time_first-new.log.txt')),
        ('baseline-time_first-linux-new', extract_fn('../log/deen-logs/e6d6_baseline-train-linux-new.log.txt')),

        ('1886', extract_fn('../log/deen-logs/1886-train.log.txt')),
        # ('1886-time_first', extract_fn('../log/deen-logs/1886-train-time_first.log.txt')),
        # ('1886-time_first-linux', extract_fn('../log/deen-logs/1886-train-time_first-linux.log.txt')),
        ('1886-time_first-new', extract_fn('../log/deen-logs/1886-train-time_first-new.log.txt')),
        ('1886-time_first-linux-new', extract_fn('../log/deen-logs/1886-train-time_first-linux-new.log.txt')),

        ('1963', extract_fn('../log/deen-logs/1963-train.log.txt')),
        # ('1963-time_first', extract_fn('../log/deen-logs/1963-train-time_first.log.txt')),
        # ('1963-time_first-linux', extract_fn('../log/deen-logs/1963-train-time_first-linux.log.txt')),
        ('1963-time_first-new', extract_fn('../log/deen-logs/1963-train-time_first-new.log.txt')),
        ('1963-time_first-linux-new', extract_fn('../log/deen-logs/1963-train-time_first-linux-new.log.txt')),
    ])

    _plot(
        e_ppl_dict, subset,
        x_range=np.arange(0, 61, 5),
        # extra_title='Default: batch_first, update_freq=32, dropout=0.3',
        style=_simple_style_fn,
    )


def main():
    init()
    # plot_deen_ppl({
    #     'subset': 'dev',
    # })
    # plot_ende_ppl({
    #     'subset': 'dev',
    # })
    plot_deen_ppl2({
        'subset': 'train',
    })


if __name__ == '__main__':
    main()
