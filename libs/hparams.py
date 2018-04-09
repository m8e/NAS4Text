#! /usr/bin/python
# -*- coding: utf-8 -*-

from argparse import Namespace

from .utils.registry_utils import camel2snake

__author__ = 'fyabc'

_HParams = {}


def register_hparams(fn_or_name):
    """

    Args:
        fn_or_name:

    Returns:

    """
    def decorator(fn, registration_name=None):
        if registration_name in _HParams:
            raise ValueError('Name {} already exists'.format(registration_name))
        _HParams[registration_name] = fn
        return fn

    if isinstance(fn_or_name, str):
        return lambda fn: decorator(fn, registration_name=fn_or_name)

    name = camel2snake(fn_or_name.__name__)
    return decorator(fn_or_name, registration_name=name)


def get_hparams(name):
    """

    Args:
        name:

    Returns:

    """
    return _HParams[name]()


@register_hparams('base')
def hparams_base():
    """Base hparams, just for test."""
    return Namespace(
        batch_size=4,
        src_seq_length=15,
        trg_seq_length=18,
        src_embedding_size=9,
        trg_embedding_size=10,
        dropout=0.1,
    )
