#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch.optim.lr_scheduler as lr_scheduler

from . import BaseLRScheduler, register_lr_scheduler

__author__ = 'fyabc'


@register_lr_scheduler('reduce_lr_on_plateau')
class ReduceLROnPlateau(BaseLRScheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self, hparams, optimizer):
        super().__init__(hparams, optimizer)
        if len(hparams.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with reduce_lr_on_plateau.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer.optimizer, patience=0, factor=hparams.lr_shrink)

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {
            'best': self.lr_scheduler.best,
            'last_epoch': self.lr_scheduler.last_epoch,
        }

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.lr_scheduler.best = state_dict['best']
        if 'last_epoch' in state_dict:
            self.lr_scheduler.last_epoch = state_dict['last_epoch']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        if val_loss is not None:
            self.lr_scheduler.step(val_loss, epoch)
        else:
            self.lr_scheduler.last_epoch = epoch
        return self.optimizer.get_lr()
