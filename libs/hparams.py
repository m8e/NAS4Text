#! /usr/bin/python
# -*- coding: utf-8 -*-

from argparse import Namespace

from .utils.registry_utils import camel2snake

__author__ = 'fyabc'

AllHParams = {}


def register_hparams(fn_or_name):
    """

    Args:
        fn_or_name:

    Returns:

    """
    def decorator(fn, registration_name=None):
        if registration_name in AllHParams:
            raise ValueError('Name {} already exists'.format(registration_name))
        AllHParams[registration_name] = fn
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
    return AllHParams[name]()


@register_hparams('base')
def hparams_base():
    """Base hparams, just for test."""

    # TODO: Add these hparams into args

    return Namespace(
        max_src_positions=15,
        max_trg_positions=18,

        src_embedding_size=9,
        trg_embedding_size=10,
        decoder_out_embedding_size=8,
        share_input_output_embedding=False,
        dropout=0.1,

        lstm_space='base',
        conv_space='base',
        attn_space='base',
    )


@register_hparams('normal')
def hparams_normal():
    """Normal hparams."""

    hparams = hparams_base()
    hparams.max_src_positions = 1024
    hparams.max_trg_positions = 1024
    hparams.src_embedding_size = 512
    hparams.trg_embedding_size = 512
    hparams.decoder_out_embedding_size = 256

    return hparams
