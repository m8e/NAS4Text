#! /usr/bin/python
# -*- coding: utf-8 -*-

import importlib
import os

from .base import BaseLRScheduler

__author__ = 'fyabc'

AllLRSchedulers = {}


def build_lr_scheduler(hparams, optimizer):
    """

    Args:
        hparams:
        optimizer:

    Returns:
        BaseLRScheduler
    """
    return AllLRSchedulers[hparams.lr_scheduler](hparams, optimizer)


def register_lr_scheduler(name):
    """Decorator to register a new LR scheduler."""

    def register_lr_scheduler_cls(cls):
        if name in AllLRSchedulers:
            raise ValueError('Cannot register duplicate LR scheduler ({})'.format(name))
        if not issubclass(cls, BaseLRScheduler):
            raise ValueError('LR Scheduler ({}: {}) must extend BaseLRScheduler'.format(name, cls.__name__))
        AllLRSchedulers[name] = cls
        return cls

    return register_lr_scheduler_cls


# automatically import any Python files in the optim/lr_scheduler/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('.' + module, package=__name__)
