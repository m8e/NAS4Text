#! /usr/bin/python
# -*- coding: utf-8 -*-

# TODO: Support multi-card parallel training.

import logging
import math

import torch as th
import torch.nn as nn

from libs.models.darts_child_net import DartsChildNet
from libs.criterions import build_criterion
from libs.utils.args import get_darts_search_args
from libs.utils import main_utils as mu
from libs.utils.meters import StopwatchMeter
from libs.child_trainer import ChildTrainer

__author__ = 'fyabc'


class DartsTrainer(ChildTrainer):
    def __init__(self, hparams, model, criterion):
        super().__init__(hparams, model, criterion)

        # [NOTE]: In DARTS, optimizer is fixed to Momemtum SGD, and lr scheduler is fixed to CosineAnnealingLR.

    # TODO: Implement train_step and valid_step, add DARTS training method and arch dumping.


def darts_search_main(hparams):
    components = mu.main_entry(hparams, train=True, net_code=False)
    hparams.net_code_file = 'darts'
    datasets = components['datasets']

    logging.info('Building model')
    model = DartsChildNet(hparams)
    # TODO: Add ParalleledChildNet here.
    # [NOTE]: In DARTS, criterion is fixed to CrossEntropyLoss.
    criterion = build_criterion(hparams, datasets.source_dict, datasets.target_dict)
    mu.logging_model_criterion(model, criterion)

    trainer = DartsTrainer(hparams, model, criterion)
    mu.logging_training_stats(hparams)

    epoch, batch_offset = mu.prepare_checkpoint(hparams, trainer)

    # [NOTE]: Dummy train step omitted.

    # Train until the learning rate gets too small
    max_epoch = hparams.max_epoch or math.inf
    max_update = hparams.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > hparams.min_lr and epoch <= max_epoch:
        # Train for one epoch
        # TODO: Replace with my train
        mu.train(hparams, trainer, datasets, epoch, batch_offset)

        # Evaluate on validate set
        if epoch % hparams.validate_interval == 0:
            for k, subset in enumerate(hparams.valid_subset.split(',')):
                # TODO: Replace with my validate
                val_loss = mu.validate(hparams, trainer, datasets, subset, epoch)
                if k == 0:
                    # Only use first validation loss to update the learning schedule
                    lr = trainer.lr_step(epoch, val_loss)

                    # save checkpoint
                    if not hparams.no_save:
                        # TODO: Replace with my save_checkpoint (or not?)
                        mu.save_checkpoint(trainer, hparams, epoch, 0, val_loss)
        else:
            lr = trainer.lr_step(epoch)

        epoch += 1
        batch_offset = 0

        if trainer.get_num_updates() >= max_update:
            break
    train_meter.stop()
    logging.info('Training done in {:.1f} seconds'.format(train_meter.sum))


def main(args=None):
    hparams = get_darts_search_args(args)
    darts_search_main(hparams)


if __name__ == '__main__':
    main()
