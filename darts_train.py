#! /usr/bin/python
# -*- coding: utf-8 -*-

# TODO: Support multi-card parallel training.

import logging

import torch as th
import torch.nn as nn

from libs.models.darts_child_net import DartsChildNet
from libs.utils.args import get_darts_search_args
from libs.utils.main_utils import main_entry

__author__ = 'fyabc'


class DartsTrainer:
    def __init__(self, hparams, model, criterion, optimizer, datasets):
        self.hparams = hparams
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.datasets = datasets

    def get_lr(self):
        pass


def darts_search_main(hparams):
    components = main_entry(hparams, train=True, net_code=False)
    hparams.net_code_file = 'darts'
    datasets = components['datasets']

    logging.info('Building model')
    model = DartsChildNet(hparams)
    # TODO: Add ParalleledChildNet here.
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()

    # TODO: More optimizers?
    optimizer = th.optim.SGD(
        model.parameters(),
        hparams.lr[0],
        momentum=hparams.momentum,
        weight_decay=hparams.weight_decay,
    )

    # TODO: More lr schedulers?
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, hparams.max_update, eta_min=hparams.min_lr,
    )


def main(args=None):
    hparams = get_darts_search_args(args)
    darts_search_main(hparams)


if __name__ == '__main__':
    main()
