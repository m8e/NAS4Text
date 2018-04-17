#! /usr/bin/python
# -*- coding: utf-8 -*-

from torch.optim.optimizer import Optimizer, required

from . import BaseOptimizer, register_optimizer

__author__ = 'fyabc'


class _NAG(Optimizer):
    def __init__(self, params, lr=required, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, lr_old=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']
            lr_old = group.get('lr_old', lr)
            lr_correct = lr / lr_old

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = d_p.clone().zero_()

                buf = param_state['momentum_buffer']

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                p.data.add_(momentum * momentum * lr_correct, buf)
                p.data.add_(-(1 + momentum) * lr, d_p)

                buf.mul_(momentum * lr_correct).add_(-lr, d_p)

            group['lr_old'] = lr

        return loss


@register_optimizer('nag')
class NAG(BaseOptimizer):
    def __init__(self, hparams, params):
        super().__init__(hparams, params)
        self._optimizer = _NAG(params, **self.optimizer_config)

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
            'momentum': self.hparams.momentum,
            'weight_decay': self.hparams.weight_decay,
        }
