#! /usr/bin/python
# -*- coding: utf-8 -*-

import math

from . import BaseLRScheduler, register_lr_scheduler

__author__ = 'fyabc'


@register_lr_scheduler('cosine')
class CosineSchedule(BaseLRScheduler):
    """Set up the lr based on an cosine schedule

    Support the proposal in
    Loshchilov, Ilya, and Frank Hutter. "Sgdr: Stochastic gradient descent with warm restarts." arXiv preprint arXiv:1608.03983 (2016).

    During warmup:

      lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
      lr = lrs[update_num]

    After warmup:

      lr = 0.5 * decay_factor *(1 + cos (\pi * ((num_updates - args.warmup_updates) % args.cosine_cycle_steps)/ args.cosine_cycle_steps))
    where
      decay_factor = args.lr * sqrt(args.warmup_updates)
    """

    def __init__(self, hparams, optimizer):
        super().__init__(hparams, optimizer)
        if len(hparams.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with cosine.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = hparams.lr[0]
        if hparams.warmup_init_lr < 0:
            hparams.warmup_init_lr = warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - hparams.warmup_init_lr) / hparams.warmup_updates

        # then, decay prop. to the update number
        self.decay_factor = warmup_end_lr

        # initial learning rate
        self.lr = hparams.warmup_init_lr
        self.min_lr = hparams.min_lr * 1.05
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # [NOTE]: "warmup-updates" and "warmup-init-lr" have been added by "inversed_sqrt" scheduler.
        parser.add_argument('--cosine-cycle-steps', default=30000, type=int, metavar='ST',
                            help='the number of optimization steps to perform one cosine cycle')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        """Update the learning rate after each update."""
        if num_updates < self.hparams.warmup_updates:
            self.lr = self.hparams.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.min_lr + \
                      0.5 * (self.decay_factor - self.min_lr) * \
                      (1.0 + math.cos(math.pi * (
                              ((num_updates - self.hparams.warmup_updates) % self.hparams.cosine_cycle_steps + 0.0) /
                              self.hparams.cosine_cycle_steps)))
        self.optimizer.set_lr(self.lr)
        return self.lr
