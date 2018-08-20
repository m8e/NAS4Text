#! /usr/bin/python
# -*- coding: utf-8 -*-

import collections
from contextlib import contextmanager
import itertools
import logging
import math
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from libs.models.block_child_net import BlockChildNet
from libs.models.nao_child_net import NAOController
from libs.optimizers import build_optimizer
from libs.optimizers.lr_schedulers import build_lr_scheduler
from libs.criterions import build_criterion
from libs.hparams import hparams_env
from libs.utils import main_utils as mu
from libs.utils.meters import StopwatchMeter, AverageMeter
from libs.utils.progress_bar import build_progress_bar
from libs.utils.paths import get_model_path
from libs.child_trainer import ChildTrainer

__author__ = 'fyabc'


class NAOTrainer(ChildTrainer):
    # [NOTE]: Flag. Train different arch on different GPUs.
    ArchDist = False

    def __init__(self, hparams, criterion):
        super().__init__(hparams, None, criterion)
        # [NOTE]: Model is a "shared" model here.
        self.controller = NAOController(hparams).cuda()
        self.model = self.controller.shared_weights
        self.arch_pool = []
        self.arch_pool_prob = None
        self.eval_arch_pool = []
        self.performance_pool = []
        self.num_gpus = hparams.distributed_world_size

    def new_model(self, net_code, device=None, cuda=True):
        result = BlockChildNet(net_code, self.hparams, self.controller)
        if cuda:
            result = result.cuda(device)
        return result

    def init_arch_pool(self):
        if self.arch_pool:
            return
        num_seed_arch = self.hparams.num_seed_arch
        self.arch_pool_prob = None

        self.arch_pool = self.controller.generate_arch(num_seed_arch)

        print('###')
        print(a.blocks for a in self.arch_pool[:10])

    def _sample_arch_from_pool(self):
        prob = self.arch_pool_prob
        if prob is None:
            pool_size = len(self.arch_pool)
            index = th.zeros([], dtype=th.int64).random_(0, pool_size).item()
        else:
            index = th.multinomial(prob).item()
        return self.arch_pool[index]

    def train_children(self, datasets):
        # self.set_seed()

        eval_freq = self.hparams.child_eval_freq

        if self.single_gpu:
            arch = self._sample_arch_from_pool()
            child = self.new_model(arch)

            # Train the child model for some epochs.
            with self.child_train_env(child):
                self._init_meters()
                for epoch in range(eval_freq):
                    # TODO: Need test here.
                    mu.train(self.hparams, self, datasets, epoch, 0)

            return

        if self.ArchDist:
            # # Random sample one arch per card to train.
            # for device in range(self.num_gpus):
            #     arch = self._sample_arch_from_pool()
            #     child = self.new_model(arch, device)
            #
            #     # TODO: How to distributed training on all GPU cards async?
            raise NotImplementedError('Arch dist multi-gpu training not supported yet')
        else:
            raise NotImplementedError('Non-arch dist multi-gpu training not supported yet')

    def eval_children(self, datasets):
        """Eval all arches in the pool."""
        eval_freq = self.hparams.child_eval_freq

        val_loss_list = []
        val_acc_list = []
        valid_time = StopwatchMeter()
        for arch in self.arch_pool:
            child = self.new_model(arch)
            # TODO: Need test.
            with self.child_train_env(child):
                val_loss = mu.validate(self.hparams, self, datasets, 'dev', eval_freq)
                val_loss_list.append(val_loss)
                val_acc_list.append(0.0)    # TODO: Also compute the valid accuracy (BLEU).

        valid_time.stop()
        logging.info('''\
Evaluation on valid data
Totally validated {} architectures
loss={:<6f}
valid_accuracy={:<8.6f}
secs={:<10.2f}'''.format(
            len(self.arch_pool), np.mean(val_loss_list), np.mean(val_acc_list),
            valid_time.sum,
        ))
        return val_acc_list

    def _get_train_itr(self, datasets, seed, epoch=1, batch_offset=0):
        hparams = self.hparams
        # The max number of positions can be different for train and valid
        # e.g., RNNs may support more positions at test time than seen in training
        max_positions_train = (
            min(hparams.max_src_positions, self.get_model().max_encoder_positions()),
            min(hparams.max_trg_positions, self.get_model().max_decoder_positions()),
        )

        # Initialize dataloader, starting at batch_offset
        itr = datasets.train_dataloader(
            hparams.train_subset,
            max_tokens=hparams.max_tokens,
            max_sentences=hparams.max_sentences,
            max_positions=max_positions_train,
            seed=seed,
            epoch=epoch,
            sample_without_replacement=hparams.sample_without_replacement,
            sort_by_source_size=(epoch <= hparams.curriculum),
            shard_id=hparams.distributed_rank,
            num_shards=hparams.distributed_world_size,
        )

        next(itertools.islice(itr, batch_offset, batch_offset), None)

        return itr

    def _get_valid_itr(self, datasets, subset='dev'):
        hparams = self.hparams

        # Initialize dataloader
        max_positions_valid = (
            self.get_model().max_encoder_positions(),
            self.get_model().max_decoder_positions(),
        )
        itr = datasets.eval_dataloader(
            subset,
            max_tokens=hparams.max_tokens,
            max_sentences=hparams.max_sentences_valid,
            max_positions=max_positions_valid,
            skip_invalid_size_inputs_valid_test=hparams.skip_invalid_size_inputs_valid_test,
            descending=True,  # largest batch first to warm the caching allocator
            shard_id=hparams.distributed_rank,
            num_shards=hparams.distributed_world_size,
        )

        return itr

    def train_step(self, sample, update_params=True):
        raise RuntimeError('This method must not be called')

    @contextmanager
    def child_train_env(self, model):
        optimizer = build_optimizer(self.hparams, model.parameters())
        lr_scheduler = build_lr_scheduler(self.hparams, optimizer)
        logging.info('Creating child train environment')
        logging.info('Child optimizer: {}'.format(optimizer.__class__.__name__))
        logging.info('Child LR Scheduler: {}'.format(lr_scheduler.__class__.__name__))

        old_model = self.model
        old_optimizer = self.optimizer
        old_lr_scheduler = self.lr_scheduler
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        try:
            yield
        finally:
            self.model = old_model
            self.optimizer = old_optimizer
            self.lr_scheduler = old_lr_scheduler
            logging.info('Trainer restored')

    def child_train_step(self, sample, update_params=True):
        return super().train_step(sample, update_params=update_params)

    def controller_train_step(self):
        pass

    def controller_generate_step(self):
        pass

    # TODO: Train step, etc.


def nao_search_main(hparams):
    components = mu.main_entry(hparams, train=True, net_code='nao')
    datasets = components['datasets']

    logging.info('Building model')
    criterion = build_criterion(hparams, datasets.source_dict, datasets.target_dict)
    trainer = NAOTrainer(hparams, criterion)
    model = trainer.get_model()
    mu.logging_model_criterion(model, criterion)
    mu.logging_training_stats(hparams)

    max_ctrl_step = hparams.max_ctrl_step or math.inf
    ctrl_step = 1
    train_meter = StopwatchMeter()
    train_meter.start()
    while ctrl_step <= max_ctrl_step:
        # Train child model.
        trainer.init_arch_pool()

        trainer.train_children(datasets)

        # Evaluate seed arches.
        valid_acc_list = trainer.eval_children(datasets)

        # Output arches and evaluated error rate.
        old_arches = trainer.arch_pool
        old_arches_perf = [1.0 - i for i in valid_acc_list]

        # Train encoder-predictor-decoder.
        trainer.controller_train_step()

        # Generate new arches.
        trainer.controller_generate_step()

        ctrl_step += 1
    train_meter.stop()
    logging.info('Training done in {:.1f} seconds'.format(train_meter.sum))


def add_nao_search_args(parser):
    group = parser.add_argument_group('NAO search options')

    group.add_argument('--max-ctrl-step', default=1000, type=int,
                       help='Number of max controller steps in arch search, default is %(default)s')
    group.add_argument('--num-seed-arch', default=1000, type=int,
                       help='Number of seed arches, default is %(default)s')
    group.add_argument('--num-nodes', default=4, type=int,
                       help='Number of nodes in one block, default is %(default)s')
    group.add_argument('--cell-op-space', default='default',
                       help='The search space of cell ops, default is %(default)r')
    group.add_argument('--num-encoder-layers', default=2, type=int,
                       help='Number of encoder layers in arch search, default is %(default)s')
    group.add_argument('--num-decoder-layers', default=2, type=int,
                       help='Number of decoder layers in arch search, default is %(default)s')
    group.add_argument('--child-eval-freq', default=10, type=int,
                       help='Number of epochs to run in between evaluations, default is %(default)s')

    # TODO

    return group


def get_nao_search_args(args=None):
    import argparse
    from libs.utils import args as utils_args
    parser = argparse.ArgumentParser(description='NAO search Script.')
    utils_args.add_general_args(parser)
    utils_args.add_dataset_args(parser, train=True)
    utils_args.add_hparams_args(parser)
    utils_args.add_train_args(parser)
    utils_args.add_distributed_args(parser)
    utils_args.add_checkpoint_args(parser)
    add_nao_search_args(parser)

    parsed_args = parser.parse_args(args)

    utils_args.parse_extra_options(parsed_args)

    return parsed_args


def main(args=None):
    hparams = get_nao_search_args(args)
    nao_search_main(hparams)


if __name__ == '__main__':
    main()
