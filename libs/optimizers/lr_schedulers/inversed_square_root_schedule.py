#! /usr/bin/python
# -*- coding: utf-8 -*-

from . import BaseLRScheduler, register_lr_scheduler

__author__ = 'fyabc'


@register_lr_scheduler('inverse_sqrt')
class InverseSquareRootSchedule(BaseLRScheduler):
    """Decay the LR based on the inverse square root of the update number.

        We also support a warmup phase where we linearly increase the learning rate
        from some initial learning rate (`--warmup-init-lr`) until the configured
        learning rate (`--lr`). Thereafter we decay proportional to the number of
        updates, with a decay factor set to align with the configured learning rate.

        During warmup:

          lrs = torch.linspace(args.warmup_init_lr, args.lr, args.warmup_updates)
          lr = lrs[update_num]

        After warmup:

          lr = decay_factor / sqrt(update_num)

        where

          decay_factor = args.lr * sqrt(args.warmup_updates)
        """

    def __init__(self, hparams, optimizer):
        super().__init__(hparams, optimizer)
        if len(hparams.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = hparams.lr[0]
        if hparams.warmup_init_lr < 0:
            hparams.warmup_init_lr = warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - hparams.warmup_init_lr) / hparams.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * hparams.warmup_updates ** 0.5

        # initial learning rate
        self.lr = hparams.warmup_init_lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.hparams.warmup_updates:
            self.lr = self.hparams.warmup_init_lr + num_updates * self.lr_step
        else:
            self.lr = self.decay_factor * num_updates ** -0.5
        self.optimizer.set_lr(self.lr)
        return self.lr
