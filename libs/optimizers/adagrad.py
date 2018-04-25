#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch.optim as optim

from . import BaseOptimizer, register_optimizer

__author__ = 'fyabc'


@register_optimizer('adagrad')
class Adagrad(BaseOptimizer):
    def __init__(self, hparams, params):
        super().__init__(hparams, params)
        self._optimizer = optim.Adagrad(params, **self.optimizer_config)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.hparams.lr[0],
            'weight_decay': self.hparams.weight_decay,
        }
