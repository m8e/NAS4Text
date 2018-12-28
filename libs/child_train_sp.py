#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Single process training functions."""

import collections
import itertools
import logging
import math
import os

import torch as th

from .utils import main_utils as mu
from .utils.data_processing import LanguageDatasets
from .models.child_net_base import ParalleledChildNet
from .criterions import build_criterion
from .child_trainer import ChildTrainer
from .utils.common import get_net_type
from .utils.paths import get_model_path
from .utils.meters import StopwatchMeter, AverageMeter
from .utils.progress_bar import build_progress_bar

from .utils.debug_utils import load_fairseq_checkpoint

__author__ = 'fyabc'


def single_process_main(hparams, datasets=None):
    """Main entry for training the child network.

    Can be called by user or the teacher model.

    Args:
        hparams: Hyper-parameters passed to the child trainer.
        datasets: Preload datasets, may be used to share dataset when called by teacher model.

    Returns:
        The child trainer instance.
    """

    # TODO: Like fairseq-py, add multiprocessing and distributed training.

    components = mu.main_entry(hparams, datasets=datasets, train=True)
    net_code = components['net_code']
    datasets = components['datasets']

    # Build model and criterion
    model = get_net_type(net_code)(net_code, hparams)
    model = ParalleledChildNet(model, output_device=hparams.device_id)
    criterion = build_criterion(hparams, datasets.source_dict, datasets.target_dict)
    mu.logging_model_criterion(model, criterion, logging_params=False)

    # model = load_fairseq_checkpoint('F:/Users/v-yaf/GitProjects/NAS4Text/models/fairseq_models/e6d6_baseline.pt', model)

    # Build trainer
    trainer = ChildTrainer(hparams, model, criterion)
    mu.logging_training_stats(hparams)

    epoch, batch_offset = mu.prepare_checkpoint(hparams, trainer)

    # Send a dummy batch to warm the caching allocator
    dummy_batch = datasets.get_dataset('train').get_dummy_batch(hparams.max_tokens, trainer.get_model().max_positions())
    # dummy_batch = datasets.get_dataset('test').get_dummy_batch(hparams.max_tokens, trainer.get_model().max_positions())     # [DEBUG]
    trainer.dummy_train_step(dummy_batch)

    # Train until the learning rate gets too small
    max_epoch = hparams.max_epoch or math.inf
    max_update = hparams.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > hparams.min_lr and epoch <= max_epoch:
        # Train for one epoch
        mu.train(hparams, trainer, datasets, epoch, batch_offset)

        # Evaluate on validate set
        if epoch % hparams.validate_interval == 0:
            for k, subset in enumerate(hparams.valid_subset.split(',')):
                val_loss = mu.validate(hparams, trainer, datasets, subset, epoch)
                if k == 0:
                    # Only use first validation loss to update the learning schedule
                    lr = trainer.lr_step(epoch, val_loss)

                    # save checkpoint
                    if not hparams.no_save:
                        mu.save_checkpoint(trainer, hparams, epoch, 0, val_loss)
        else:
            lr = trainer.lr_step(epoch)

        epoch += 1
        batch_offset = 0

        if trainer.get_num_updates() >= max_update:
            break
    train_meter.stop()
    logging.info('Training done in {:.1f} seconds'.format(train_meter.sum))
