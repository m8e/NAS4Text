#! /usr/bin/python
# -*- coding: utf-8 -*-

from contextlib import contextmanager
import copy
import itertools
import logging
import math
import os

import numpy as np
import torch as th

from libs.models.block_child_net import BlockChildNet
from libs.models.nao_child_net import NAOController
from libs.optimizers import build_optimizer
from libs.optimizers.lr_schedulers import build_lr_scheduler
from libs.criterions import build_criterion
from libs.utils import main_utils as mu
from libs.utils.meters import StopwatchMeter
from libs.utils.paths import get_model_path
from libs.child_trainer import ChildTrainer
from libs.child_generator import ChildGenerator

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
        self.branch_length = 2  # TODO: Change this hparams.

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

    def _sample_arch_from_pool(self):
        # TODO: Sample by model size?
        prob = self.arch_pool_prob
        if prob is None:
            pool_size = len(self.arch_pool)
            index = th.zeros([], dtype=th.int64).random_(0, pool_size).item()
        else:
            index = th.multinomial(prob).item()
        return self.arch_pool[index]

    def train_children(self, datasets):
        logging.info('Training children, arch pool size = {}'.format(len(self.arch_pool)))
        eval_freq = self.hparams.child_eval_freq

        if self.single_gpu:
            arch = self._sample_arch_from_pool()
            child = self.new_model(arch)

            # Train the child model for some epochs.
            with self.child_train_env(child):
                logging.info('Number of child model parameters: {}'.format(child.num_parameters()))
                print('Architecture:', arch.blocks['enc1'], arch.blocks['dec1'], sep='\n\t')

                self._init_meters()
                for epoch in range(1, eval_freq + 1):
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

    def eval_children(self, datasets, ret_acc=True):
        """Eval all arches in the pool."""
        eval_freq = self.hparams.child_eval_freq

        val_loss_list = []
        val_acc_list = []
        valid_time = StopwatchMeter()
        for arch in self.arch_pool:
            child = self.new_model(arch)
            with self.child_train_env(child, train=False):
                val_loss = mu.validate(self.hparams, self, datasets, 'dev', eval_freq)
            val_loss_list.append(val_loss)

            # TODO: How to compute BLEU? Generation too slow?
            if ret_acc:
                val_translations = self._generate_on_dev(child, datasets)
                val_acc_list.append(0.0)
            else:
                val_acc_list.append(math.nan)

        valid_time.stop()
        logging.info('''\
Evaluation on valid data: totally validated {} architectures
Metrics: loss={:<6f}, valid_accuracy={:<8.6f}, secs={:<10.2f}'''.format(
            len(self.arch_pool), np.mean(val_loss_list), np.mean(val_acc_list),
            valid_time.sum,
        ))
        if ret_acc:
            return val_acc_list
        else:
            return val_loss_list

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

    @contextmanager
    def child_train_env(self, model, train=True):
        logging.info('Creating child {} environment'.format('train' if train else 'valid'))
        if train:
            optimizer = build_optimizer(self.hparams, model.parameters())
            lr_scheduler = build_lr_scheduler(self.hparams, optimizer)
            logging.info('Child optimizer: {}'.format(optimizer.__class__.__name__))
            logging.info('Child LR Scheduler: {}'.format(lr_scheduler.__class__.__name__))

        old_model = self.model
        self.model = model
        if train:
            old_optimizer = self.optimizer
            old_lr_scheduler = self.lr_scheduler
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        try:
            yield
        finally:
            self.model = old_model
            if train:
                self.optimizer = old_optimizer
                self.lr_scheduler = old_lr_scheduler
            logging.info('Trainer restored')

    def _generate_on_dev(self, model, datasets, device=None):
        logging.info('Generate on dev set')
        generator = ChildGenerator(
            self.hparams, datasets, [model], subset='dev', quiet=True, output_file=None,
            use_task_maxlen=False, maxlen_a=0, maxlen_b=200, max_tokens=None, max_sentences=128,
            beam=5, lenpen=1.2,
        )
        generator.cuda(device)
        translated_strings = generator.greedy_decoding()

        return translated_strings

    def _parse_arch_to_seq(self, arch):
        # TODO
        pass

    def _normalized_perf(self, perf_list):
        max_val, min_val = np.max(perf_list), np.min(perf_list)
        return [(v - min_val) / (max_val - min_val) for v in perf_list]

    def controller_train_step(self, old_arches, old_arches_perf):
        encoder_input = [self._parse_arch_to_seq(arch) for arch in old_arches]
        encoder_target = self._normalized_perf(old_arches_perf)
        decoder_target = copy.deepcopy(encoder_target)

        # TODO: Train the epd.

    def controller_generate_step(self, old_arches):
        old_arches = old_arches[:self.hparams.num_remain_top]
        new_arches = []
        predict_lambda = 0
        topk_arches = [self._parse_arch_to_seq(arch) for arch in old_arches[:self.hparams.num_pred_top]]

        while len(new_arches) + len(old_arches) < self.hparams.num_seed_arch:
            # TODO: Generate new arches.
            predict_lambda += 1
            pass

    # TODO: Train step, etc.


def nao_search_main(hparams):
    RetAcc = False  # FIXME: Flag. Return accuracy or loss. Standard is True.

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
        logging.info('Training step {}'.format(ctrl_step))
        trainer.set_seed(ctrl_step)

        # Train child model.
        trainer.init_arch_pool()

        trainer.train_children(datasets)

        # Evaluate seed arches.
        valid_acc_list = trainer.eval_children(datasets, ret_acc=RetAcc)

        # Output arches and evaluated error rate.
        old_arches = trainer.arch_pool
        if RetAcc:
            # Error rate list.
            old_arches_perf = [1.0 - i for i in valid_acc_list]
        else:
            old_arches_perf = valid_acc_list[:]

        # Sort old arches.
        old_arches_sorted_indices = np.argsort(old_arches_perf)
        old_arches = np.array(old_arches)[old_arches_sorted_indices].tolist()
        old_arches_perf = np.array(old_arches_perf)[old_arches_sorted_indices].tolist()

        # Save old arches in order.
        save_arches(hparams, ctrl_step, old_arches, old_arches_perf)

        # Train encoder-predictor-decoder.
        trainer.controller_train_step(old_arches, old_arches_perf)

        # Generate new arches.
        trainer.controller_generate_step(old_arches)

        ctrl_step += 1
    train_meter.stop()
    logging.info('Training done in {:.1f} seconds'.format(train_meter.sum))


def save_arches(hparams, ctrl_step, arches, arches_perf):
    import json

    save_dir = get_model_path(hparams)
    net_code_list = [n.original_code for n in arches]

    def _save(code_filename, perf_filename):
        full_code_filename = os.path.join(save_dir, code_filename)
        with open(full_code_filename, 'w', encoding='utf-8') as f:
            json.dump(net_code_list, f)
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
    group.add_argument('--cell-op-space', default='no-zero',
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


def main(args=None):
    hparams = get_nao_search_args(args)
    nao_search_main(hparams)


if __name__ == '__main__':
    main()
