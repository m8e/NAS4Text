#! /usr/bin/python
# -*- coding: utf-8 -*-

"""NAO utils."""

import copy
import logging
import os

import torch as th
import torch.utils.data as th_data

from .paths import get_model_path
from . import common

__author__ = 'fyabc'


def make_ctrl_dataloader(arch_seqs, perf, batch_size, shuffle, sos_id):
    # Build inputs, and collect them into batches. [batch_size, source_length]
    encoder_input = arch_seqs
    encoder_target = perf
    source_length = len(encoder_input)
    # Create decoder input.
    # FIXME: A problem: Why use [1:], not [:-1]?
    decoder_input = th.cat(
        (th.LongTensor([[sos_id]] * source_length),
         th.LongTensor(encoder_input)[:, 1:]),
        dim=1)
    decoder_target = copy.copy(encoder_input)
    dataset = th_data.TensorDataset(
        th.LongTensor(encoder_input), th.Tensor(encoder_target),
        th.LongTensor(decoder_input), th.LongTensor(decoder_target),
    )

    loader = th_data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


def prepare_ctrl_sample(sample, evaluation=False):
    encoder_input, encoder_target, decoder_input, decoder_target = sample
    sample = {
        'encoder_input': encoder_input,
        'encoder_target': encoder_target,
        'decoder_input': decoder_input,
        'decoder_target': decoder_target,
    }
    return common.make_variable(sample, cuda=True, volatile=evaluation)


def save_arches(hparams, ctrl_step, arches, arches_perf):
    import json

    save_dir = get_model_path(hparams)
    net_code_list = [n.original_code for n in arches]

    def _save(code_filename, perf_filename):
        full_code_filename = os.path.join(save_dir, code_filename)
        with open(full_code_filename, 'w', encoding='utf-8') as f:
            json.dump(net_code_list, f, indent=4)
            logging.info('Save arches into {} (epoch {})'.format(full_code_filename, ctrl_step))
        full_perf_filename = os.path.join(save_dir, perf_filename)
        with open(full_perf_filename, 'w', encoding='utf-8') as f:
            for perf in arches_perf:
                print(perf, file=f)
            logging.info('Save performance into {} (epoch {})'.format(full_perf_filename, ctrl_step))

    _save('arch_pool{}.json'.format(ctrl_step), 'arch_perf{}.txt'.format(ctrl_step))
    _save('arch_pool_last.json', 'arch_perf_last.txt')


def add_nao_search_args(parser):
    group = parser.add_argument_group('NAO search options')

    group.add_argument('--max-ctrl-step', default=1000, type=int,
                       help='Number of max controller steps in arch search, default is %(default)s')
    group.add_argument('--num-seed-arch', default=1000, type=int,
                       help='Number of seed arches, default is %(default)s')
    group.add_argument('--num-remain-top', default=500, type=int,
                       help='Number of remaining top-k best arches, default is %(default)s')
    group.add_argument('--num-pred-top', default=100, type=int,
                       help='Number of top-k best arches used in prediction, default is %(default)s')
    group.add_argument('--num-nodes', default=4, type=int,
                       help='Number of nodes in one block, default is %(default)s')
    group.add_argument('--cell-op-space', default='only-attn-no-zero',
                       help='The search space of cell ops, default is %(default)r')
    group.add_argument('--num-encoder-layers', default=2, type=int,
                       help='Number of encoder layers in arch search, default is %(default)s')
    group.add_argument('--num-decoder-layers', default=2, type=int,
                       help='Number of decoder layers in arch search, default is %(default)s')
    group.add_argument('--child-eval-freq', default=10, type=int,
                       help='Number of epochs to run in between evaluations, default is %(default)s')
    group.add_argument('--child-eval-batch-size', default=32, type=int,
                       help='Number of batch size in evaluations, default is %(default)s')

    # Controller hyper-parameters.
    group.add_argument('--ctrl-train-epochs', default=1000, type=int,
                       help='Controller training epochs, default is %(default)s')
    group.add_argument('--ctrl-batch-size', default=100, type=int,
                       help='Controller batch size, default is %(default)s')
    group.add_argument('--ctrl-enc-length', default=None, type=int,
                       help='Controller encoder length, default same as source length')
    group.add_argument('--ctrl-dec-length', default=None, type=int,
                       help='Controller decoder length, default same as source length')
    group.add_argument('--ctrl-enc-vocab-size', default=10, type=int,
                       help='Controller encoder embedding size, default is %(default)s')
    group.add_argument('--ctrl-dec-vocab-size', default=10, type=int,
                       help='Controller decoder embedding size, default is %(default)s')
    group.add_argument('--ctrl-enc-emb-size', default=96, type=int,
                       help='Controller encoder embedding size, default is %(default)s')
    group.add_argument('--ctrl-enc-hidden-size', default=96, type=int,
                       help='Controller encoder hidden size, default is %(default)s')
    group.add_argument('--ctrl-mlp-hidden-size', default=200, type=int,
                       help='Controller predictor hidden size, default is %(default)s')
    group.add_argument('--ctrl-dec-hidden-size', default=96, type=int,
                       help='Controller decoder hidden size, default is %(default)s')
    group.add_argument('--ctrl-num-encoder-layers', default=1, type=int,
                       help='Number of controller encoder layers, default is %(default)s')
    group.add_argument('--ctrl-num-mlp-layers', default=0, type=int,
                       help='Number of controller predictor layers, default is %(default)s')
    group.add_argument('--ctrl-num-decoder-layers', default=1, type=int,
                       help='Number of controller decoder layers, default is %(default)s')
    group.add_argument('--ctrl-enc-dropout', default=0.1, type=float,
                       help='Controller encoder dropout, default is %(default)s')
    group.add_argument('--ctrl-mlp-dropout', default=0.4, type=float,
                       help='Controller predictor dropout, default is %(default)s')
    group.add_argument('--ctrl-dec-dropout', default=0.0, type=float,
                       help='Controller decoder dropout, default is %(default)s')
    group.add_argument('--ctrl-weight-decay', default=1e-4, type=float,
                       help='Controller weight decay, default is %(default)s')
    group.add_argument('--ctrl-weighted-loss', action='store_true', default=False,
                       help='Controller using weighted loss, default is False')
    group.add_argument('--ctrl-trade-off', default=0.8, type=float,
                       help='Controller trade off between losses (weight of EP loss), default is %(default)s')
    group.add_argument('--ctrl-lr', default=0.001, type=float,
                       help='Controller learning rate, default is %(default)s')
    group.add_argument('--ctrl-optimizer', default='adam', type=str,
                       help='Controller optimizer, default is %(default)s')
    group.add_argument('--ctrl-clip-norm', default=5.0, type=float,
                       help='Controller clip norm, default is %(default)s')

    # TODO

    return group


def get_nao_search_args(args=None):
    import argparse
    from libs.utils import args as utils_args
    parser = argparse.ArgumentParser(description='NAO search Script.')
    utils_args.add_general_args(parser)
    utils_args.add_dataset_args(parser, train=True, gen=True)
    utils_args.add_hparams_args(parser)
    utils_args.add_train_args(parser)
    utils_args.add_distributed_args(parser)
    utils_args.add_checkpoint_args(parser)
    utils_args.add_generation_args(parser)  # For BLEU scorer.
    add_nao_search_args(parser)

    parsed_args = parser.parse_args(args)

    utils_args.parse_extra_options(parsed_args)

    return parsed_args
