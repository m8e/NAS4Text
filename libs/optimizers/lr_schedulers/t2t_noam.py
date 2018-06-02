#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th

from . import BaseLRScheduler, register_lr_scheduler

__author__ = 'fyabc'


@register_lr_scheduler('t2t_noam')
class T2TNoamSchedule(BaseLRScheduler):
    """The T2T noam learning rate scheduler."""

    def __init__(self, hparams, optimizer):
        super().__init__(hparams, optimizer)
        if len(hparams.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with t2t_noam.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = hparams.lr[0]
        if hparams.warmup_init_lr < 0:
            hparams.warmup_init_lr = warmup_end_lr

        self.warmup_updates = float(self.hparams.warmup_updates * th.cuda.device_count())
        self.decay_factor = 5000.0 * hparams.src_embedding_size ** -0.5

        self.lr_base = self.hparams.lr[0]
        self.lr = self.lr_base
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # [NOTE]: All args have been added by "inversed_sqrt" scheduler.

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""

        self.lr = self.lr_base * self.decay_factor * min(
            (num_updates + 1) * self.warmup_updates ** -1.5,
            (num_updates + 1) ** -0.5,
        )
        self.optimizer.set_lr(self.lr)
        return self.lr
