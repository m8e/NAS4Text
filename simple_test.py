#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import logging

import numpy as np
import torch as th
from torch.autograd import Variable

from libs.hparams import get_hparams
from libs.child_net import ChildNet
from libs.layers.net_code import NetCodeEnum

__author__ = 'fyabc'


def get_args(args=None):
    parser = argparse.ArgumentParser(description='Simple Test Script.')

    parser.add_argument('-H', '--hparams-set', dest='hparams_set', type=str, default='base')
    parser.add_argument('-T', '--task', dest='task', type=str, default='test')

    group_hparams = parser.add_argument_group('HParams Options', description='Options that set hyper-parameters.')
    group_hparams.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=None)
    group_hparams.add_argument('--src-seq-length', dest='src_seq_length', type=int, default=None)
    group_hparams.add_argument('--trg-seq-length', dest='trg_seq_length', type=int, default=None)
    group_hparams.add_argument('--src-emb-size', dest='src_embedding_size', type=int, default=None)
    group_hparams.add_argument('--trg-emb-size', dest='trg_embedding_size', type=int, default=None)
    group_hparams.add_argument('--lstm-space', dest='lstm_space', type=str, default=None)
    group_hparams.add_argument('--conv-space', dest='conv_space', type=str, default=None)
    group_hparams.add_argument('--attn-space', dest='attn_space', type=str, default=None)

    parsed_args = parser.parse_args(args)
    base_hparams = get_hparams(parsed_args.hparams_set)

    for name, value in base_hparams.__dict__.items():
        if getattr(parsed_args, name, None) is None:
            setattr(parsed_args, name, value)

    return parsed_args


def get_sample_dataset(hparams):
    from libs.tasks import get_task
    task = get_task(hparams.task)

    return [
        [
            Variable(th.from_numpy(np.random.randint(
                1, task.SourceVocabSize, size=(hparams.batch_size, hparams.src_seq_length - 2), dtype='int64'))),
            Variable(th.from_numpy(np.random.randint(
                1, task.TargetVocabSize, size=(hparams.batch_size, hparams.trg_seq_length - 2), dtype='int64'))),
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
