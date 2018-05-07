#! /usr/bin/python
# -*- coding: utf-8 -*-

# TODO: Support user-defined hparams.

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
        share_src_trg_embedding=False,
        dropout=0.1,

        lstm_space='base',
        conv_space='base',
        attn_space='base',

        # Feed-forward size of self-attention layers (aka "filter_size" in T2T)
        attn_d_ff=2048,

        # Candidates: dot_product, fairseq
        enc_dec_attn_type='dot_product',

        # Candidates: none, layer, batch, noam
        norm_type='layer',
        norm_epsilon=1e-6,
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


@register_hparams('fairseq_de_en_iwslt')
def hparams_fairseq_de_en_iwslt():
    """HParams of de-en iwslt, copied from fairseq-py."""

    hparams = hparams_normal()

    hparams.conv_space = 'large'
    hparams.src_embedding_size = 256
    hparams.trg_embedding_size = 256
    hparams.decoder_out_embedding_size = 256

    return hparams


@register_hparams('fairseq_attn')
def hparams_fairseq_attn_de_en_iwslt():
    """HParams of fairseq-py on de-en iwslt, and use fairseq attention."""

    hparams = hparams_fairseq_de_en_iwslt()
    hparams.enc_dec_attn_type = 'fairseq'

    return hparams


@register_hparams('transformer_de_en_iwslt')
def hparams_transformer_de_en_iwslt():
    """HParams of de-en iwslt, copied from T2T."""

    hparams = hparams_normal()

    hparams.max_src_positions = 256
    hparams.max_trg_positions = 256
    hparams.src_embedding_size = 256
    hparams.trg_embedding_size = 256
    hparams.decoder_out_embedding_size = 256
    hparams.attn_d_ff = 1024

    return hparams


@register_hparams('transformer_attn')
def hparams_transformer_attn_de_en_iwslt():
    """HParams of T2T on de-en iwslt, and use fairseq attention."""

    hparams = hparams_transformer_de_en_iwslt()
    hparams.enc_dec_attn_type = 'fairseq'

    return hparams


@register_hparams('transformer_share_emb')
def hparams_transformer_de_en_iwslt_share_emb():
    """HParams of T2T on de-en iwslt, and use shared embedding and softmax."""

    hparams = hparams_transformer_de_en_iwslt()
    hparams.share_input_output_embedding = True

    return hparams


@register_hparams('transformer_share3')
def hparams_transformer_de_en_iwslt_share3():
    """HParams of T2T on de-en iwslt, and use shared source embedding, target embedding and softmax."""

    hparams = hparams_transformer_de_en_iwslt()
    hparams.share_input_output_embedding = True
    hparams.share_src_trg_embedding = True

    return hparams
