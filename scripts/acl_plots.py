#! /usr/bin/python
# -*- coding: utf-8 -*-

# [NOTE]: Use matplotlib-1.5.3.

# FIXME: The PA and hamming in iteration 1 curve select from
#   "log/plot-logs/nao-train-deen-iwslt14-standalone-nao-aug4.txt",
#   not the original training log (which is not found)!
#
# FIXME: The PA and hamming curve trained on iter3 and iter4 arch-bleu data on GCRAZGDW141.

import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

RootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LogPath = os.path.join(RootPath, 'log', 'plot-logs')

__author__ = 'fyabc'


def init():
    print(matplotlib.__version__)

    text = {
        'usetex': True,
    }
    font = {
        # 'weight': 'bold',
        'size': 72,
        # 'family': 'sans-serif',
        # 'sans-serif': ['Helvetica'],
    }
    lines = {
        'linewidth': 12,
        'markersize': 20,
    }
    # plt.rc('text', **text)
    plt.rc('font', **font)
    plt.rc('lines', **lines)


def set_formatter(extra_fmt_x='', extra_fmt_y='', axes=None):
    ax = plt.gca() if axes is None else axes
    ax.xaxis.set_major_formatter(StrMethodFormatter(rf'${{x{extra_fmt_x}}}$'))
    ax.yaxis.set_major_formatter(StrMethodFormatter(rf'${{x{extra_fmt_y}}}$'))


def bar_autolabel(bars, ax=None, extra_fmt=':.2f', dy=0.05):
    if ax is None:
        ax = plt.gca()
    for rect in bars:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., height + dy,
                '{{{}}}'.format(extra_fmt).format(height), ha='center', va='bottom')


def plot_mean_dev_bleu():
    iterations = [1, 2, 3]
    # epd_pa = [0.654, 0.766, 0.839]
    mean_dev_bleu = [30.26, 32.29, 33.72]

    # # Iteration 4
    # iterations.append(4)
    # epd_pa.append(0.862)
    # mean_dev_bleu.append(33.27)

    bars = plt.bar(iterations, mean_dev_bleu, width=0.6, align='center', color='royalblue')
    bar_autolabel(bars)
    plt.xlabel(r'Iteration $l$')
    plt.ylabel(r'$BLEU$ on dev set')

    plt.xticks(iterations)
    plt.grid(linestyle='--')
    plt.gca().set_axisbelow(True)
    plt.ylim(ymin=30, ymax=35)

    set_formatter()

    plt.show()


def plot_pa_hamming_in1iteration(pa: bool = True):
    """Plot the pairwise accuracy and the hamming distance in one iteration."""
    ctrl_epochs = []
    test_pa = []
    test_hd = []
    with open(os.path.join(LogPath, 'nao-train-deen-iwslt14-standalone-nao-aug4.txt'), 'r', encoding='utf-8') as f:
        for line in f:
            if '| ctrl eval (test) |' in line:
                words = line.split()
                ctrl_epochs.append(int(words[11].lstrip('0')))
                test_pa.append(float(words[15]))
                test_hd.append(float(words[19]))

    # # #epochs -> #arhictectures
    # arch_per_epoch = 4030
    # ctrl_epochs = [e * arch_per_epoch for e in ctrl_epochs]

    if pa:
        plt.plot(ctrl_epochs, test_pa, color='blue')
        plt.xlabel(r'Number of epochs for Encoder-Predictor-Decoder training', fontsize=48)
        plt.ylabel(r'$acc_f$')

        plt.grid(linestyle='--')
        # plt.xlim(left=0, right=100)

        set_formatter(':.0f', ':.2f')
    else:
        plt.plot(ctrl_epochs, test_hd, 'r-o')
        plt.xlabel(r'Number of epochs for Encoder-Predictor-Decoder training', fontsize=48)
        plt.ylabel(r'$Dist_D$')

        plt.grid(linestyle='--')
        plt.xlim(left=1, right=10)

        set_formatter(':.0f', ':.2f')

    plt.show()


def plot_pa_over_train_size():
    """Plot the PA-#(training size) curve."""
    num_train_archs = [0, 50, 100, 150, 200, 250, 300]
    pa = [0.5, 0.662142, 0.711712, 0.737237, 0.764264, 0.788288, 0.831832]

    xmax = 310
    scale_x = False

    if scale_x:
        num_train_archs = [0, 150, 300, 450, 600, 750, 900]
        xmax = 910

    plt.plot(num_train_archs, pa, 'b-o')
    plt.xlabel('Number of models for training', fontsize=60)
    plt.ylabel(r'$acc_f$')

    plt.grid(linestyle='--')
    plt.xlim(xmin=-5, xmax=xmax)
    plt.ylim(ymin=0.5, ymax=0.86)

    if scale_x:
        plt.xticks(range(150, xmax, 150))

    set_formatter(':.0f', ':.2f')

    plt.show()


def plot_hamming_over_train_size():
    """Plot the PA-#(training size) curve."""
    num_train_archs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    hamming = [25.0, 19.108108, 15.135135, 4.378378, 3.567568, 3.297297,
               1.783784, 1.027027, 0.459459, 0.000000, 0.000000]

    xmax = 102

    plt.plot(num_train_archs, hamming, 'r-o')
    plt.xlabel('Number of models for training', fontsize=60)
    plt.ylabel(r'$Dist_D$')

    plt.grid(linestyle='--')
    plt.xlim(xmin=-1, xmax=xmax)
    plt.ylim(ymin=0.0, ymax=26.0)

    set_formatter(':.0f', ':.2f')

    plt.show()


def main():
    init()
    # plot_mean_dev_bleu()
    # plot_pa_hamming_in1iteration(pa=True)
    # plot_pa_hamming_in1iteration(pa=False)
    # plot_pa_over_train_size()
    plot_hamming_over_train_size()


if __name__ == '__main__':
    main()
