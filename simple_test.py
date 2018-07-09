#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
from collections import namedtuple

import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from libs.utils.args import get_args
from libs.child_net import ChildNet
from libs.utils.search_space import LayerTypes

__author__ = 'fyabc'


def get_sample_dataset(hparams):
    from libs.tasks import get_task
    task = get_task(hparams.task)

    Batch = namedtuple('Batch', ['src_tokens', 'src_lengths', 'trg_tokens', 'trg_lengths'])

    result = []
    for _ in range(10):
        bs = [4, 6][np.random.choice(2, 1)[0]]
        src_tokens = Variable(th.from_numpy(np.random.randint(
            1, task.SourceVocabSize,
            size=(bs, np.random.randint(1, hparams.max_src_positions)),
            dtype='int64')).cuda())
        src_lengths = Variable(th.LongTensor(bs).fill_(src_tokens.size()[1]).cuda())
        trg_tokens = Variable(th.from_numpy(np.random.randint(
            1, task.TargetVocabSize,
            size=(bs, np.random.randint(1, hparams.max_trg_positions)),
            dtype='int64')).cuda())
        trg_lengths = Variable(th.LongTensor(bs).fill_(trg_tokens.size()[1]).cuda())
        result.append(Batch(src_tokens, src_lengths, trg_tokens, trg_lengths))

    return result


def main(args=None):
    logging.basicConfig(
        format='{levelname}:{message}',
        level=logging.INFO,
        style='{',
    )

    hparams = get_args(args)

    net_code = [
        [
            [LayerTypes.LSTM, 0, 1],
            [LayerTypes.Convolutional, 2, 1, 0],
            [LayerTypes.Attention, 0],
        ],
        [
            [LayerTypes.LSTM, 1, 0],
        ]
    ]

    net = ChildNet(net_code, hparams=hparams)
    net = net.cuda()

    print('Network:', net)
    print()

    dataset = get_sample_dataset(hparams)

    optimizer = optim.Adadelta(net.parameters())

    for epoch in range(10):
        for batch in dataset:
            optimizer.zero_grad()

            pred_trg_probs = net(*batch)
            logging.debug('')

            target = batch.trg_tokens
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
