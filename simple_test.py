#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

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
        level=logging.INFO,
        style='{',
    )

    hparams = get_args(args)

    net_code = [
        [
            [NetCodeEnum.LSTM, 0, 1],
            [NetCodeEnum.Convolutional, 2, 1, 2],
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

    optimizer = optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(10):
        for batch in dataset:
            optimizer.zero_grad()

            pred_trg_probs = net(*batch)
            logging.debug('')

            target = batch[1]
            loss = F.cross_entropy(
                pred_trg_probs.view(-1, pred_trg_probs.size(-1)),
                target.view(-1),
                size_average=False,
                ignore_index=0,
            )
            loss.backward()
            print('Loss = {}'.format(loss.data[0]))

            optimizer.step()

            corrects = target == pred_trg_probs.max(dim=-1)[1]
            print('Argmax error rate:', (1.0 - corrects.float().sum() / corrects.nelement()).data[0])


if __name__ == '__main__':
    main()
