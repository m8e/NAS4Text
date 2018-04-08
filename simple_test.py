#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse

import torch as th
import torch.nn as nn
from torch.autograd import Variable

from libs.child_net import ChildNet
from libs.layers.net_code import NetCodeEnum

__author__ = 'fyabc'


def get_args():
    parser = argparse.ArgumentParser(description='Simple Test Script.')

    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=4)
    parser.add_argument('-l', '--seq-length', dest='seq_length', type=int, default=15)
    parser.add_argument('--in-emb-size', dest='input_embedding_size', type=int, default=10)
    parser.add_argument('-T', '--task', dest='task', type=str, default='de_en_iwslt')

    return parser.parse_args()


def get_sample_dataset(hparams):
    return [
        Variable(th.randn(hparams.batch_size, hparams.seq_length, hparams.input_embedding_size))
        for _ in range(10)
    ]


def main():
    hparams = get_args()

    net_code = [
        [
            [NetCodeEnum.LSTM, 2, 1],
        ],
        [
            [NetCodeEnum.LSTM, 1, 0],
        ]
    ]

    net = ChildNet(net_code, hparams=hparams)

    dataset = get_sample_dataset(hparams)

    for epoch in range(5):
        for batch in dataset:
            print('Input a tensor of shape', batch.shape)

            net.zero_grad()

            output = net(batch)
            print('Produce a tensor of shape', output.shape)


if __name__ == '__main__':
    main()
