#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse

from ..hparams import get_hparams

__author__ = 'fyabc'


def get_args(args=None):
    parser = argparse.ArgumentParser(description='Simple Test Script.')

    parser.add_argument('-H', '--hparams-set', dest='hparams_set', type=str, default='base')
    parser.add_argument('-T', '--task', dest='task', type=str, default='test')

    group_hparams = parser.add_argument_group('HParams Options', description='Options that set hyper-parameters.')
    group_hparams.add_argument('-b', '--batch-size', dest='batch_size', type=int, default=None)
    group_hparams.add_argument('--src-seq-length', dest='src_seq_length', type=int, default=None)
    group_hparams.add_argument('--trg-seq-length', dest='trg_seq_length', type=int, default=None)
    group_hparams.add_argument('--src-emb-size', dest='src_embedding_size', type=int, default=None)
    group_hparams.add_argument('--trg-emb-size', dest='trg_embedding_size', type=int, default=None)
    group_hparams.add_argument('--lstm-space', dest='lstm_space', type=str, default=None)
    group_hparams.add_argument('--conv-space', dest='conv_space', type=str, default=None)
    group_hparams.add_argument('--attn-space', dest='attn_space', type=str, default=None)

    parsed_args = parser.parse_args(args)
    base_hparams = get_hparams(parsed_args.hparams_set)

    for name, value in base_hparams.__dict__.items():
        if getattr(parsed_args, name, None) is None:
            setattr(parsed_args, name, value)

    return parsed_args
