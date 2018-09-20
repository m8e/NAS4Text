#! /usr/bin/python
# -*- coding: utf-8 -*-

# TODO: Support user-defined hparams.

from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy

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


_sentinel = None


@contextmanager
def hparams_env(hparams, **kwargs):
    new_hparams = deepcopy(hparams)
    for k, v in kwargs.items():
        setattr(new_hparams, k, v)
    try:
        yield new_hparams
    finally:
        pass


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
        embed_scale=False,

        dropout=0.1,
        ppp_dropout=0.1,        # Dropout for layer pre-post-processing.
        attention_dropout=0.1,
        ffn_dropout=0.1,        # Dropout for FFN layer after each attention layer.

        # Model structure options.
        apply_grad_mul=False,
        connect_src_emb=False,
        connect_trg_emb=False,
        enc_output_fc=False,
        dec_output_fc=False,
        attn_linear_bias=False,
        enc_out_norm=False,
        dec_out_norm=False,
        enc_learned_pos=True,
        dec_learned_pos=True,

        # This define the search space of three layer types.
        lstm_space='base',
        conv_space='base',
        attn_space='base',

        # Feed-forward size of self-attention layers (aka "filter_size" in T2T)
        attn_d_ff=2048,

        # Candidates: dot_product, fairseq
        enc_dec_attn_type='dot_product',

        # Candidates: none, layer, batch, noam
        norm_type='layer',
        norm_epsilon=1e-5,

        # About initializer
        # Candidates: original, uniform_init_scaling, kaitao, fairseq,
        initializer='uniform_unit_scaling',
        initializer_gain=1.0,

        # About training
        lr='0.25',
        momentum=0.99,
        weight_decay=0.0,
        clip_norm=25,

        # About block child net.
        # Candidates: concat, add, last
        block_combine_op='concat',
        # Only combine "no out" nodes or not (combine all output nodes instead).
        block_combine_no_outs=False,

        # About NAS.

        # # Stub for DARTS
        # # Cell op space.
        # cell_op_space='default'
        # # Number of nodes in one block.
        # num_nodes=4,
        # # Number of nodes combined into output in one block.
        # num_output_nodes=4,
        # # Number of encoder and decoder layers in arch search.
        # num_encoder_layers=6,
        # num_decoder_layers=6,
        # # Portion of training data.
        # train_portion=0.5,
        # # Clip threshold of gradients in arch search.
        # arch_clip_norm=10.0,
        # # Arch learning rate.
        # arch_lr=3e-4,
        # # Arch search optimizer.
        # arch_optimizer='adam',
        # # Arch Adam betas.
        # arch_adam_betas='(0.5, 0.999)',
        # # Arch weight decay.
        # arch_weight_decay=1e-3,

        # # Stub for NAO
        # # Number of max controller steps.
        # max_ctrl_step=1000,
        # # Number of seed arches.
        # num_seed_arch=1000,
        # # Number of remaining top-k best arches.
        # num_remain_top=500,
        # # Number of nodes in one block.
        # num_nodes=4,
        # # Cell op space.
        # cell_op_space='default'
        # # Number of encoder and decoder layers in arch search.
        # num_encoder_layers=6,
        # num_decoder_layers=6,
        # # Number of epochs to run in between evaluations.
        # child_eval_freq=10,
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

    hparams.clip_norm = 2.5

    return hparams


@register_hparams('share3')
def hparams_share3():
    """HParams that share 3 embeddings and softmax."""

    hparams = hparams_normal()
    hparams.src_embedding_size = 256
    hparams.trg_embedding_size = 256
    hparams.share_input_output_embedding = True
    hparams.share_src_trg_embedding = True

    return hparams


@register_hparams('fairseq_de_en_iwslt')
def hparams_fairseq_de_en_iwslt():
    """HParams of de-en iwslt, copied from fairseq-py."""

    hparams = hparams_normal()

    hparams.conv_space = 'large'
    hparams.src_embedding_size = 256
    hparams.trg_embedding_size = 256
    hparams.decoder_out_embedding_size = 256

    hparams.clip_norm = 0.1
    hparams.dropout = 0.2
    hparams.ppp_dropout = 0.2
    hparams.attn_dropout = 0.2
    hparams.ffn_dropout = 0.2

    return hparams


@register_hparams('fairseq_share3')
def hparams_fairseq_de_en_iwslt_share3():
    """HParams of fairseq-py on de-en iwslt, and use shared source embedding, target embedding and softmax."""

    hparams = hparams_fairseq_de_en_iwslt()
    hparams.share_input_output_embedding = True
    hparams.share_src_trg_embedding = True

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

    hparams.max_src_positions = 1024
    hparams.max_trg_positions = 1024
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


@register_hparams('transformer_share3_kt')
def hparams_transformer_de_en_iwslt_share3_kaitao():
    hparams = hparams_transformer_de_en_iwslt_share3()

    hparams.clip_norm = 0.1
    hparams.dropout = 0.2
    hparams.ppp_dropout = 0.2
    hparams.attn_dropout = 0.2
    hparams.ffn_dropout = 0.2
    hparams.initializer = 'kaitao'

    return hparams


@register_hparams('bpe2_transformer_share3_kt')
def hparams_transformer_de_en_iwslt_bpe2_share3_kaitao():
    hparams = hparams_transformer_de_en_iwslt_share3()

    hparams.clip_norm = 0.1
    hparams.dropout = 0.1
    hparams.ppp_dropout = 0.1
    hparams.attn_dropout = 0.1
    hparams.ffn_dropout = 0.1
    hparams.initializer = 'kaitao'
    hparams.share_src_trg_embedding = False
    hparams.share_input_output_embedding = False

    return hparams


@register_hparams('bpe2_transformer_kt_bias')
def hparams_transformer_de_en_iwslt_bpe2_kaitao_bias():
    hparams = hparams_transformer_de_en_iwslt_bpe2_share3_kaitao()

    hparams.initializer = 'kaitao_wn'
    hparams.attn_linear_bias = True

    return hparams


@register_hparams('bpe2_transformer_l2_best')
def hparams_transformer_de_en_iwslt_bpe2_e2d2_best():
    """The best hparams of en-de-iwslt14 transformer e2d2 now."""
    hparams = hparams_transformer_de_en_iwslt_bpe2_kaitao_bias()

    # # Copy inherited hparams here.
    # hparams.max_src_positions = 1024
    # hparams.max_trg_positions = 1024
    # hparams.src_embedding_size = 256
    # hparams.trg_embedding_size = 256
    # hparams.decoder_out_embedding_size = 256
    # hparams.attn_d_ff = 1024
    # hparams.clip_norm = 0.1
    # hparams.dropout = 0.1
    # hparams.ppp_dropout = 0.1
    # hparams.attn_dropout = 0.1
    # hparams.ffn_dropout = 0.1
    # hparams.share_src_trg_embedding = False
    # hparams.share_input_output_embedding = False
    # hparams.initializer = 'kaitao_wn'

    hparams.dec_output_fc = True
    hparams.attn_linear_bias = False
    hparams.enc_out_norm = True
    hparams.dec_out_norm = True

    return hparams


@register_hparams('bpe2_transformer_fairseq')
def hparams_transformer_de_en_iwslt_bpe2_fairseq():
    hparams = hparams_normal()

    hparams.initializer = 'fairseq'

    hparams.dropout = 0.2
    hparams.attention_dropout = 0.0
    hparams.ffn_dropout = 0.0
    hparams.ppp_dropout = hparams.dropout

    hparams.max_src_positions = 1024
    hparams.max_trg_positions = 1024
    hparams.src_embedding_size = 256
    hparams.trg_embedding_size = 256
    hparams.decoder_out_embedding_size = 256
    hparams.share_input_output_embedding = True
    hparams.attn_d_ff = 512
    hparams.attn_linear_bias = True
    hparams.enc_learned_pos = False
    hparams.dec_learned_pos = False
    hparams.embed_scale = True

    hparams.block_combine_op = 'last'

    return hparams


@register_hparams('de_en_iwslt_darts')
def hparams_de_en_iwslt_bpe2_darts():
    hparams = hparams_transformer_de_en_iwslt_bpe2_fairseq()

    hparams.block_combine_op = 'add'

    return hparams


@register_hparams('de_en_iwslt_darts_dp')
def hparams_de_en_iwslt_bpe2_darts_dropout_add_02():
    hparams = hparams_de_en_iwslt_bpe2_darts()

    hparams.dropout = 0.4
    hparams.attention_dropout = 0.2
    hparams.ffn_dropout = 0.2
    hparams.ppp_dropout = hparams.dropout

    return hparams


@register_hparams('de_en_iwslt_add_no_outs')
def hparams_de_en_iwslt_add_no_outs():
    hparams = hparams_transformer_de_en_iwslt_bpe2_fairseq()

    hparams.block_combine_op = 'add'
    hparams.block_combine_no_outs = True

    return hparams


@register_hparams('de_en_iwslt_nao')
def hparams_de_en_iwslt_nao():
    hparams = hparams_de_en_iwslt_add_no_outs()
    hparams.cell_op_space = 'only-attn'     # [NOTE]: This is dummy now, change it in future.

    return hparams


@register_hparams('de_en_iwslt_nao_dp')
def hparams_de_en_iwslt_bpe2_nao_dropout_add_02():
    hparams = hparams_de_en_iwslt_nao()

    hparams.dropout = 0.4
    hparams.attention_dropout = 0.2
    hparams.ffn_dropout = 0.2
    hparams.ppp_dropout = hparams.dropout

    return hparams
