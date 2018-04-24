#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch.optim.lr_scheduler as lr_scheduler

from . import BaseLRScheduler, register_lr_scheduler

__author__ = 'fyabc'


@register_lr_scheduler('fixed')
class FixedSchedule(BaseLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, hparams, optimizer):
        super().__init__(hparams, optimizer)
        self.lr_scheduler = lr_scheduler.LambdaLR(
            self.optimizer.optimizer, self.anneal)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                            help='force annealing at specified epoch')

    def anneal(self, epoch):
        lrs = self.hparams.lr
        if self.hparams.force_anneal is None or epoch < self.hparams.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # anneal based on lr_shrink
            next_lr = lrs[-1] * self.hparams.lr_shrink ** (epoch + 1 - self.hparams.force_anneal)
        return next_lr / lrs[0]  # correct for scaling from LambdaLR

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        self.lr_scheduler.step(epoch)
        return self.optimizer.get_lr()
