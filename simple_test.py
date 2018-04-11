#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
from torch.autograd import Variable

from libs.utils.args import get_args
from libs.child_net import ChildNet
from libs.layers.net_code import NetCodeEnum

__author__ = 'fyabc'


def get_sample_dataset(hparams):
    from libs.tasks import get_task
    task = get_task(hparams.task)

    return [
        [
            Variable(th.from_numpy(np.random.randint(
                1, task.SourceVocabSize,
                size=(hparams.batch_size, np.random.randint(1, hparams.src_seq_length)),
                dtype='int64'))),
            Variable(th.from_numpy(np.random.randint(
                1, task.TargetVocabSize,
                size=(hparams.batch_size, np.random.randint(1, hparams.trg_seq_length)),
                dtype='int64'))),
        ] for _ in range(10)]


def main(args=None):
    logging.basicConfig(
        format='{levelname}:{message}',
        level=logging.DEBUG,
        style='{',
    )

    hparams = get_args(args)

    net_code = [
        [
            [NetCodeEnum.LSTM, 2, 1],
            [NetCodeEnum.Convolutional, 0, 1, 2],
            [NetCodeEnum.Attention, 0]
        ],
        [
            [NetCodeEnum.LSTM, 1, 0],
        ]
    ]

    net = ChildNet(net_code, hparams=hparams)

    print('Network:', net)
    print()

    dataset = get_sample_dataset(hparams)

    for epoch in range(2):
        for batch in dataset:
            net.zero_grad()

            output = net(*batch)
            logging.debug('')


if __name__ == '__main__':
    main()
