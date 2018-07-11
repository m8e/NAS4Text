#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
from collections import namedtuple

import numpy as np
import torch as th
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from libs.layers.net_code import NetCode
from libs.utils.args import get_args
from libs.utils.common import get_net_type
from libs.utils.search_space import LayerTypes
from libs.utils.main_utils import main_entry

__author__ = 'fyabc'


Cuda = True


def get_sample_dataset(hparams):
    from libs.tasks import get_task
    task = get_task(hparams.task)

    Batch = namedtuple('Batch', ['src_tokens', 'src_lengths', 'trg_tokens', 'trg_lengths'])

    result = []
    for _ in range(10):
        bs = [4, 6][np.random.choice(2, 1)[0]]

        src_tokens_data = th.from_numpy(np.random.randint(
            1, task.SourceVocabSize,
            size=(bs, np.random.randint(1, hparams.max_src_positions)),
            dtype='int64'))
        src_lengths_data = th.LongTensor(bs).fill_(src_tokens_data.size()[1])
        trg_tokens_data = th.from_numpy(np.random.randint(
            1, task.TargetVocabSize,
            size=(bs, np.random.randint(1, hparams.max_trg_positions)),
            dtype='int64'))
        trg_lengths_data = th.LongTensor(bs).fill_(trg_tokens_data.size()[1])

        if Cuda:
            src_tokens_data, src_lengths_data, trg_tokens_data, trg_lengths_data = \
                src_tokens_data.cuda(), src_lengths_data.cuda(), trg_tokens_data.cuda(), trg_lengths_data.cuda()

        src_tokens = Variable(src_tokens_data)
        src_lengths = Variable(src_lengths_data)
        trg_tokens = Variable(trg_tokens_data)
        trg_lengths = Variable(trg_lengths_data)
        result.append(Batch(src_tokens, src_lengths, trg_tokens, trg_lengths))

    return result


AllNetCode = {
    'default': NetCode({
        "Global": {},
        "Layers": [
            [
                [LayerTypes.LSTM, 0, 1],
                [LayerTypes.Convolutional, 2, 1, 0],
                [LayerTypes.Attention, 0],
            ],
            [
                [LayerTypes.LSTM, 1, 0],
            ],
        ],
    }),
    'block': NetCode({
        "Type": "BlockChildNet",
        "Global": {},
        "Layers": [
            [
                # TODO
            ],
            [

            ],
        ],
    })
}


def main(args=None, net_code_key='default'):
    logging.basicConfig(
        format='{levelname}:{message}',
        level=logging.INFO,
        style='{',
    )

    hparams = get_args(args)

    main_entry(hparams, net_code=False, train=False, load_datasets=False)

    net_code = AllNetCode.get(net_code_key, AllNetCode['default'])

    net = get_net_type(net_code)(net_code, hparams=hparams)
    if Cuda:
        net = net.cuda()

    print('Network:', net)
    print()

    dataset = get_sample_dataset(hparams)

    optimizer = optim.Adadelta(net.parameters())

    for epoch in range(10):
        for batch in dataset:
            optimizer.zero_grad()

            net_output = net(*batch)
            pred_trg_probs = net.get_normalized_probs(net_output, log_probs=True)
            logging.debug('')

            target = batch.trg_tokens
            loss = F.nll_loss(
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
    main(['-H', 'bpe2_transformer_kt_bias'], net_code_key='default')
