#! /usr/bin/python
# -*- coding: utf-8 -*-

"""NAO utils."""

from bisect import bisect
import copy
import logging
import os

import numpy as np
import torch as th
import torch.utils.data as th_data

from .paths import get_model_path
from . import common

__author__ = 'fyabc'


def make_ctrl_dataloader(arch_seqs, perf, batch_size, shuffle, sos_id):
    # Build inputs, and collect them into batches. [batch_size, source_length]
    encoder_input = copy.copy(arch_seqs)

    encoder_target = perf
    source_length = len(encoder_input)
    # Create decoder input. Right shift the encoder input.
    decoder_input = th.cat(
        (th.LongTensor([[sos_id]] * source_length),
         th.LongTensor(encoder_input)[:, :-1]),
        dim=1)
    decoder_target = copy.copy(encoder_input)

    if perf is None:
        return make_tensor_dataloader([
            th.LongTensor(encoder_input),
            th.LongTensor(decoder_input), th.LongTensor(decoder_target),
        ], batch_size=batch_size, shuffle=shuffle)

    return make_tensor_dataloader([
        th.LongTensor(encoder_input), th.Tensor(encoder_target),
        th.LongTensor(decoder_input), th.LongTensor(decoder_target),
    ], batch_size=batch_size, shuffle=shuffle)


def make_tensor_dataloader(tensor_list, batch_size, shuffle):
    dataset = th_data.TensorDataset(*tensor_list)
    loader = th_data.DataLoader(dataset, batch_size, shuffle)
    return loader


def prepare_ctrl_sample(sample, evaluation=False, perf=True):
    if perf:
        encoder_input, encoder_target, decoder_input, decoder_target = sample
        sample = {
            'encoder_input': encoder_input,
            'encoder_target': encoder_target,
            'decoder_input': decoder_input,
            'decoder_target': decoder_target,
        }
    else:
        encoder_input, decoder_input, decoder_target = sample
        sample = {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'decoder_target': decoder_target,
        }
    return common.make_variable(sample, cuda=True, volatile=evaluation)


def pairwise_accuracy(la, lb):
    n = len(la)
    assert n == len(lb)
    total, count = 0, 0

    for i in range(n):
        for j in range(i + 1, n):
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            total += 1
    return float(count) / total


def hamming_distance(la, lb):
    n = len(la)
    assert n == len(lb)

    def _distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c

    dis = 0
    for line1, line2 in zip(la, lb):
        dis += _distance(line1, line2)
    return float(dis) / n


def save_arches(hparams, ctrl_step, arches, arches_perf=None, after_gen=False):
    import json

    _after_gen = ', after generate' if after_gen else ''
    _n = len(arches)

    save_dir = get_model_path(hparams)
    net_code_list = [n.original_code for n in arches]

    def _save(code_filename, perf_filename):
        full_code_filename = os.path.join(save_dir, code_filename)
        with open(full_code_filename, 'w', encoding='utf-8') as f:
            for code in net_code_list:
                print(json.dumps(code), file=f)
            logging.info('Save {} arches into {} (epoch {}{})'.format(_n, full_code_filename, ctrl_step, _after_gen))
        if arches_perf is not None:
            full_perf_filename = os.path.join(save_dir, perf_filename)
            with open(full_perf_filename, 'w', encoding='utf-8') as f:
                for perf in arches_perf:
                    print(perf, file=f)
                logging.info('Save performance into {} (epoch {}{})'.format(full_perf_filename, ctrl_step, _after_gen))

    _save('arch_pool{}.json'.format(ctrl_step), 'arch_perf{}.txt'.format(ctrl_step))
    _save('arch_pool_last.json', 'arch_perf_last.txt')


def _swap_two_branches(block, valid_indices):
    """Swap two branches of one or more nodes in the block."""
    # Random select 1 ~ N nodes to swap.
    n_to_swap = np.random.randint(1, len(valid_indices))
    node_idx_to_swap = valid_indices[:]
    np.random.shuffle(node_idx_to_swap)
    node_idx_to_swap = node_idx_to_swap[:n_to_swap]

    # Swap the two branches of one node.
    for node_idx in node_idx_to_swap:
        node = block[node_idx]
        in1, in2, op1, op2, *extra = node
        new_node = [in2, in1, op2, op1] + extra

        block[node_idx] = new_node


def _swap_non_ancestor_nodes(block, valid_indices):
    ancestor_pairs = set()

    for index in valid_indices:
        node = block[index]
        in1, in2, *_ = node
        # TODO: Add into ancestor set

    # TODO


def _arch_augment_per_block(arch):
    """Run the augmentation both for encoder and decoder blocks, and run some wrapper code."""
    from ..layers.net_code import NetCode

    new_net_code = copy.deepcopy(arch.original_code)

    for in_encoder in [False, True]:
        block = arch.blocks['enc1'] if in_encoder else arch.blocks['dec1']
        valid_indices = [i for i in range(len(block)) if isinstance(block[i], list) and block[i][0] is not None]
        new_block = new_net_code['Blocks']['enc1'] if in_encoder else new_net_code['Blocks']['dec1']

        _swap_two_branches(new_block, valid_indices)

    return NetCode(new_net_code)


def _get_augment_rep(base_rep, bleu, sorted_list):
    return max(1, int(2 * base_rep * bisect(sorted_list, bleu) / len(sorted_list)))


def arch_augmentation(arch_list, bleu_list, augment_rep=4, focus_top=False):
    """Apply the data augmentation on the architecture list.

    Create some architectures with same semantics.

    Args:
        arch_list:
        bleu_list:
        augment_rep (int):
        focus_top (bool):

    Returns:

    """

    sorted_list = sorted(bleu_list)

    orig_len = len(bleu_list)
    for i in range(orig_len):
        arch, bleu = arch_list[i], bleu_list[i]
        rep = _get_augment_rep(augment_rep, bleu, sorted_list) if focus_top else augment_rep
        for _ in range(rep):
            new_arch = _arch_augment_per_block(arch)
            if not any(new_arch.fast_eq(a) for a in arch_list):
                arch_list.append(new_arch)
                bleu_list.append(bleu)
    print('Arch augmentation: Rep = {}, #Arches from {} to {}'.format(augment_rep, orig_len, len(arch_list)))

    return arch_list, bleu_list


def add_nao_search_args(parser):
    group = parser.add_argument_group('NAO search options')

    group.add_argument('--standalone', action='store_true', default=False,
                       help='Standalone train the controller, default is False')

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
    group.add_argument('--cell-op-space', default='only-attn',
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
    group.add_argument('--ctrl-global-keys', default='',
                       help='Comma-separated global keys to be searched, default is %(default)r')
    group.add_argument('--ctrl-enc-vocab-size', default=None, type=int,
                       help='Controller encoder embedding size, default is automatically detected')
    group.add_argument('--ctrl-dec-vocab-size', default=None, type=int,
                       help='Controller decoder embedding size, default is automatically detected')
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
    group.add_argument('--ctrl-eval-freq', default=100, type=int,
                       help='Number of epochs to run between controller evaluations, default is %(default)s')
    group.add_argument('--ctrl-save-freq', default=100, type=int,
                       help='Number of epochs to run between controller savings, default is %(default)s')
    group.add_argument('--ctrl-log-freq', default=50, type=int,
                       help='Number of steps to run between controller logging, default is %(default)s')
    group.add_argument('--ctrl-eval-train-freq', default=100, type=int,
                       help='Number of epochs to run between controller training set evaluations, '
                            'default is %(default)s')
    group.add_argument('--lambda-step', type=float, default=1.0,
                       help='Lambda step, default is %(default)r')

    # Standalone hyper-parameters.
    group.add_argument('--sa-iteration', type=int,
                       help='Iteration of standalone job')
    group.add_argument('--no-augment', action='store_false', default=True, dest='augment',
                       help='Does not apply augmentation')
    group.add_argument('--augment-rep', type=int, default=8,
                       help='Augmentation replicate number, default is %(default)r')
    group.add_argument('--focus-top', action='store_true', default=False,
                       help='Focus on top architectures')
    group.add_argument('--reload', action='store_true', default=False,
                       help='Reload old model, only run generation.')

    # TODO

    return group


def get_nao_search_args(args=None):
    import argparse
    from libs.utils import args as utils_args
    parser = argparse.ArgumentParser(description='NAO search Script.')
    utils_args.add_general_args(parser)
    utils_args.add_dataset_args(parser, train=True, gen=True)
    utils_args.add_hparams_args(parser)
    utils_args.add_extra_options_args(parser)
    utils_args.add_train_args(parser)
    utils_args.add_distributed_args(parser)
    utils_args.add_checkpoint_args(parser)
    utils_args.add_generation_args(parser)  # For BLEU scorer.
    add_nao_search_args(parser)

    parsed_args = parser.parse_args(args)

    utils_args.parse_extra_options(parsed_args)

    return parsed_args


def hparams_ppp_nao(hparams):
    hparams.ctrl_global_keys = [k for k in hparams.ctrl_global_keys.split(',') if k != '']
