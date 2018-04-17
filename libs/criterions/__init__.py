#! /usr/bin/python
# -*- coding: utf-8 -*-

import importlib
import os

from .base import BaseCriterion

__author__ = 'fyabc'


AllCriterions = {}
AllCriterionClassNames = set()


def build_criterion(hparams, src_dict, trg_dict):
    return AllCriterions[hparams.criterion](hparams, src_dict, trg_dict)


def register_criterion(name):
    """Decorator to register a new criterion."""

    def register_criterion_cls(cls):
        if name in AllCriterions:
            raise ValueError('Cannot register duplicate criterion ({})'.format(name))
        if not issubclass(cls, BaseCriterion):
            raise ValueError('Criterion ({}: {}) must extend BaseCriterion'.format(name, cls.__name__))
        if cls.__name__ in AllCriterionClassNames:
            # We use the criterion class name as a unique identifier in
            # checkpoints, so all criterions must have unique class names.
            raise ValueError('Cannot register criterion with duplicate class name ({})'.format(cls.__name__))
        AllCriterions[name] = cls
        AllCriterionClassNames.add(cls.__name__)
        return cls

    return register_criterion_cls


# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('.' + module, package=__name__)
