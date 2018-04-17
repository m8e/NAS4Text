#! /usr/bin/python
# -*- coding: utf-8 -*-

import importlib
import os

from .base import BaseOptimizer

__author__ = 'fyabc'


AllOptimizers = {}
AllOptimizerClassNames = set()


def build_optimizer(hparams, params):
    """

    Args:
        hparams:
        params:

    Returns:
        BaseOptimizer
    """
    params = filter(lambda p: p.requires_grad, params)
    return AllOptimizers[hparams.optimizer](hparams, params)


def register_optimizer(name):
    """Decorator to register a new optimizer."""

    def register_optimizer_cls(cls):
        if name in AllOptimizers:
            raise ValueError('Cannot register duplicate optimizer ({})'.format(name))
        if not issubclass(cls, BaseOptimizer):
            raise ValueError('Optimizer ({}: {}) must extend BaseOptimizer'.format(name, cls.__name__))
        if cls.__name__ in AllOptimizerClassNames:
            # We use the optimizer class name as a unique identifier in
            # checkpoints, so all optimizer must have unique class names.
            raise ValueError('Cannot register optimizer with duplicate class name ({})'.format(cls.__name__))
        AllOptimizers[name] = cls
        AllOptimizerClassNames.add(cls.__name__)
        return cls

    return register_optimizer_cls


# automatically import any Python files in the optim/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('.' + module, package=__name__)
