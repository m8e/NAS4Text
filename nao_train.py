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
import torch.nn.functional as F
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from libs.models.block_child_net import BlockChildNet
from libs.models.nao_child_net import NAOController
from libs.optimizers import build_optimizer
from libs.optimizers.lr_schedulers import build_lr_scheduler
from libs.criterions import build_criterion
from libs.utils import main_utils as mu
from libs.utils.meters import StopwatchMeter
from libs.utils.paths import get_data_path, get_model_path
from libs.utils.generator_utils import batch_bleu
from libs.utils import tokenizer
from libs.utils import nao_utils
from libs.utils import common
from libs.child_trainer import ChildTrainer
from libs.child_generator import ChildGenerator
from libs.hparams import hparams_env

__author__ = 'fyabc'

_sentinel = object()


class NAOTrainer(ChildTrainer):
    # TODO: Support continue training (override ``load/save_checkpoint``).
    # Need to save and load shared weights, arches, controller optimizer, etc.

    # [NOTE]: Flags.
    ArchDist = False            # Train different arch on different GPUs.
    GenSortByLength = False     # Sort by length in generation.
    GenMaxlenB = 100            # Max length bias in generation. (less than normal generation to avoid oom)

    def __init__(self, hparams, criterion, only_epd_cuda=False):
        # [NOTE]: Model is a "shared" model here.
        self.controller = NAOController(hparams).cuda(only_epd=only_epd_cuda, epd_device=hparams.epd_device)
        super().__init__(hparams, self.controller.shared_weights, criterion)

        self.only_epd_cuda = only_epd_cuda
        self.main_device = th.cuda.current_device()
        self.device_ids_for_gen = self._get_device_ids_for_gen()

        self.arch_pool = []
        self.arch_pool_prob = None
        self.eval_arch_pool = []
        self.performance_pool = []
        self._ref_tokens = None
        self._ref_dict = None
        self._current_child_size = 0
        self._current_grad_size = 0

        with hparams_env(
            hparams, optimizer=hparams.ctrl_optimizer,
            lr=[hparams.ctrl_lr], adam_eps=1e-8,
            weight_decay=hparams.ctrl_weight_decay,
        ) as ctrl_hparams:
            self.ctrl_optimizer = build_optimizer(ctrl_hparams, self.controller.epd.parameters())
            logging.info('Controller optimizer: {}'.format(self.ctrl_optimizer.__class__.__name__))

        # Meters.
        self._ctrl_best_pa = {
            'training': 0.00,
            'test': 0.00,
        }

    def _get_device_ids_for_gen(self):
        all_device_ids = set(range(self.hparams.distributed_world_size))
        all_device_ids.discard(self.main_device)
        all_device_ids.discard(self.hparams.epd_device)
        return sorted(all_device_ids)

    def new_model(self, net_code, device=None, cuda=True,
                  device_ids=None, output_device=_sentinel, force_single=False):
        """Create a new child model (instance of ``BlockChildNet``) from given net code and shared weights."""
        result = BlockChildNet(net_code, self.hparams, self.controller)
        if not force_single and self.hparams.distributed_world_size > 1:
            if cuda:
                if device_ids is None:
                    device_ids = list(range(self.hparams.distributed_world_size))
                if output_device is _sentinel:
                    output_device = self.hparams.device_id
                from libs.models.child_net_base import ParalleledChildNet
                result = ParalleledChildNet(
                    result, device_ids=device_ids,
                    output_device=output_device)
        else:
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
        # TODO: Sample by model size. (Except shared parts (embedding, etc.)
        prob = self.arch_pool_prob
        if prob is None:
            pool_size = len(self.arch_pool)
            index = th.zeros([], dtype=th.int64).random_(0, pool_size).item()
        else:
            index = th.multinomial(prob).item()
        # print('$select index is:', index)
        return self.arch_pool[index]

    def train_children(self, datasets):
        logging.info('Training children, arch pool size = {}'.format(len(self.arch_pool)))
        eval_freq = self.hparams.child_eval_freq

        if self.single_gpu:
            self._init_meters()
            for epoch in range(1, eval_freq + 1):
                mu.train(self.hparams, self, datasets, epoch, 0)

            # Restore shared model after training.
            self.model = self.controller.shared_weights

            return

        if self.ArchDist:
            # TODO: How to distributed training on all GPU cards async?
            raise NotImplementedError('Arch dist multi-gpu training not supported yet')
        else:
            # raise NotImplementedError('Non-arch dist multi-gpu training not supported yet')
            self._init_meters()
            for epoch in range(1, eval_freq + 1):
                mu.train(self.hparams, self, datasets, epoch, 0)

            # Restore shared model after training.
            self.model = self.controller.shared_weights

            return

    def train_step(self, sample, update_params=True):
        # [NOTE]: At each train step, sample a new arch from pool.
        arch = self._sample_arch_from_pool()
        child = self.new_model(arch)
        self.model = child
        self._current_child_size = self.model.num_parameters()
        return super().train_step(sample, update_params=update_params)

    def _get_flat_grads(self, out=None):
        # [NOTE]: Since model will be changed between updates, does not use out buffer.
        return super()._get_flat_grads(out=None)

    def eval_children(self, datasets, compute_loss=False):
        """Eval all arches in the pool."""

        self._get_ref_tokens(datasets)

        # Prepare the generator.
        generator = ChildGenerator(
            self.hparams, datasets, [self.model], subset='dev', quiet=True, output_file=None,
            use_task_maxlen=False, maxlen_a=0, maxlen_b=self.GenMaxlenB, max_tokens=None,
            max_sentences=self.hparams.child_eval_batch_size, beam=5, lenpen=1.2,
        )
        itr = generator.get_input_iter(sort_by_length=self.GenSortByLength)
        whole_gen_itr = common.cycled_data_iter(itr)

        arch_itr = self.arch_pool
        if tqdm is not None:
            arch_itr = tqdm(arch_itr)

        val_loss_list = []
        val_acc_list = []
        valid_time = StopwatchMeter()

        with th.cuda.device(self.hparams.gen_device):
            for arch in arch_itr:
                while True:
                    sample = next(whole_gen_itr)

                    # [NOTE]: Run data parallel on all GPUs except main device and epd device
                    child = self.new_model(arch, cuda=True, device_ids=self.device_ids_for_gen, output_device=None)

                    if compute_loss:
                        with self.child_env(child, train=False):
                            val_loss = mu.validate(self.hparams, self, datasets, 'dev', self.hparams.child_eval_freq)
                        val_loss_list.append(val_loss)

                    generator.models = [child]
                    generator.cuda()

                    # Skip OOM batches.
                    try:
                        # FIXME: Use beam 5 decoding here.
                        translation = generator.decoding_one_batch(sample)
                        val_bleu = batch_bleu(generator, sample['id'], translation, self._ref_tokens, self._ref_dict)
                        val_acc_list.append(val_bleu)
                    except RuntimeError as e:
                        logging.warning('Error when decoding a batch, skip this batch: {}'.format(e))
                        continue
                    else:
                        break

        # [NOTE]: In generation step, some of the shared weights are moved to other GPUs, move them back.
        if not self.only_epd_cuda:
            self.controller.shared_weights.cuda()

        valid_time.stop()
        logging.info('''\
Evaluation on valid data: totally validated {} architectures
Metrics: loss={}, valid_accuracy={:<8.6f}, secs={:<10.2f}'''.format(
            len(self.arch_pool), ':<6f'.format(np.mean(val_loss_list)) if compute_loss else '[NotComputed]',
            np.mean(val_acc_list), valid_time.sum,
        ))
        return val_acc_list

    @contextmanager
    def child_env(self, model, train=True):
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

    def _get_ref_tokens(self, datasets):
        if self._ref_tokens is not None:
            return
        dataset_dir = get_data_path(self.hparams)
        dev_filename = datasets.task.get_filename('dev', is_src_lang=False)
        dev_orig_filename = dev_filename + '.orig'  # FIXME: Hard-coding here.

        # [NOTE]: Reuse target dictionary.
        from copy import deepcopy
        dict_ = self._ref_dict = deepcopy(datasets.target_dict)
        # dict_ = self._ref_dict = dictionary.Dictionary(None, datasets.task, mode='empty')

        self._ref_tokens = []
        with open(os.path.join(dataset_dir, dev_orig_filename), 'r', encoding='utf-8') as ref_f:
            for line in ref_f:
                self._ref_tokens.append(tokenizer.Tokenizer.tokenize(line, dict_, tensor_type=th.IntTensor))

    def _parse_arch_to_seq(self, arch):
        return self.controller.parse_arch_to_seq(arch)

    def _normalized_perf(self, perf_list):
        max_val, min_val = np.max(perf_list), np.min(perf_list)
        return [(v - min_val) / (max_val - min_val) for v in perf_list]

    def _shuffle_and_split(self, a, p, test_size=0.1):
        index = np.random.permutation(len(a))
        a = [a[i] for i in index]
        p = [p[i] for i in index]
        split = int(np.floor(len(a) * test_size))
        return (a[split:], p[split:]), (a[:split], p[:split])

    def controller_train_step(self, old_arches, old_arches_perf, split_test=True):
        logging.info('Training Encoder-Predictor-Decoder')
        augment = self.hparams.augment
        augment_rep = self.hparams.augment_rep

        perf = self._normalized_perf(old_arches_perf)

        if split_test:
            train_ap, test_ap = self._shuffle_and_split(old_arches, perf, test_size=0.1)
            if augment:
                train_ap = nao_utils.arch_augmentation(
                    *train_ap, augment_rep=self.hparams.augment_rep, focus_top=self.hparams.focus_top)
            train_arches, train_bleus = train_ap
            train_ap = [self._parse_arch_to_seq(arch) for arch in train_arches], train_bleus
            test_arches, test_bleus = test_ap
            test_ap = [self._parse_arch_to_seq(arch) for arch in test_arches], test_bleus

            ctrl_dataloader = nao_utils.make_ctrl_dataloader(
                *train_ap, batch_size=self.hparams.ctrl_batch_size,
                shuffle=True, sos_id=self.controller.epd.sos_id)
            test_ctrl_dataloader = nao_utils.make_ctrl_dataloader(
                *test_ap, batch_size=self.hparams.ctrl_batch_size,
                shuffle=False, sos_id=self.controller.epd.sos_id)
        else:
            if augment:
                old_arches, perf = nao_utils.arch_augmentation(
                    old_arches, perf, augment_rep=augment_rep, focus_top=self.hparams.focus_top)
            arch_seqs = [self._parse_arch_to_seq(arch) for arch in old_arches]
            ctrl_dataloader = nao_utils.make_ctrl_dataloader(
                arch_seqs, perf, batch_size=self.hparams.ctrl_batch_size,
                shuffle=True, sos_id=self.controller.epd.sos_id)
            test_ctrl_dataloader = ctrl_dataloader

        epochs = range(1, self.hparams.ctrl_train_epochs + 1)
        if tqdm is not None:
            epochs = tqdm(list(epochs))

        step = 0
        with th.cuda.device(self.hparams.epd_device):
            for epoch in epochs:
                for epoch_step, sample in enumerate(ctrl_dataloader):
                    self.controller.epd.train()

                    sample = nao_utils.prepare_ctrl_sample(sample, evaluation=False)

                    # print('#Expected global range:', self.controller.expected_global_range())
                    # print('#Expected node range:', self.controller.expected_index_range())
                    # print('#Expected op range:', self.controller.expected_op_range(False), self.controller.expected_op_range(True))
                    # print('#encoder_input', sample['encoder_input'].shape, sample['encoder_input'][0])
                    # print('#encoder_target', sample['encoder_target'].shape, sample['encoder_target'][0])
                    # print('#decoder_input', sample['decoder_input'].shape, sample['decoder_input'][0])
                    # print('#decoder_target', sample['decoder_target'].shape, sample['decoder_target'][0])

                    # FIXME: Use ParallelModel here?
                    predict_value, logits, arch = self.controller.epd(sample['encoder_input'], sample['decoder_input'])

                    # print('#predict_value', predict_value.shape, predict_value.tolist())
                    # print('#logits', logits.shape)
                    # print('$arch', arch.shape, arch)

                    # Loss and optimize.
                    loss_1 = F.mse_loss(predict_value.squeeze(), sample['encoder_target'].squeeze())
                    logits_size = logits.size()
                    n = logits_size[0] * logits_size[1]
                    loss_2 = F.cross_entropy(logits.contiguous().view(n, -1), sample['decoder_target'].view(n))

                    loss = self.hparams.ctrl_trade_off * loss_1 + (1 - self.hparams.ctrl_trade_off) * loss_2

                    self.ctrl_optimizer.zero_grad()
                    loss.backward()
                    grad_norm = th.nn.utils.clip_grad_norm_(
                        self.controller.epd.parameters(), self.hparams.ctrl_clip_norm)
                    self.ctrl_optimizer.step()

                    # TODO: Better logging here.
                    if step % self.hparams.ctrl_log_freq == 0:
                        print('| ctrl | epoch {:03d} | step {:03d} | loss={:5.6f} '
                              '| mse={:5.6f} | cse={:5.6f} | gnorm={:5.6f}'.format(
                                epoch, step, loss.data, loss_1.data, loss_2.data, grad_norm,
                                ))

                    step += 1

                # TODO: Add evaluation and controller model saving here.
                if epoch % self.hparams.ctrl_eval_freq == 0:
                    if ctrl_dataloader is not test_ctrl_dataloader:
                        self.controller_eval_step(test_ctrl_dataloader, epoch, subset='test')
                        # [NOTE]: Can omit eval on training set to speed up.
                        if epoch % self.hparams.ctrl_eval_train_freq == 0:
                            self.controller_eval_step(ctrl_dataloader, epoch, subset='training')
                    else:
                        self.controller_eval_step(test_ctrl_dataloader, epoch, subset='training')

    def controller_eval_step(self, ctrl_dataloader, epoch, subset='training'):
        model = self.controller.epd
        ground_truth_perf_list = []
        ground_truth_arch_seq_list = []
        predict_value_list = []
        arch_seq_list = []

        time = StopwatchMeter()
        time.start()

        def _safe_extend(l, v):
            v = v.data.squeeze().tolist()
            if isinstance(v, (float, int)):
                l.append(v)
            else:
                l.extend(v)

        with th.cuda.device(self.hparams.epd_device):
            for step, sample in enumerate(ctrl_dataloader):
                model.eval()
                sample = nao_utils.prepare_ctrl_sample(sample, evaluation=True)
                predict_value, logits, arch = model(sample['encoder_input'])    # target_variable=None
                _safe_extend(predict_value_list, predict_value)
                arch_seq_list.extend(arch)
                _safe_extend(ground_truth_perf_list, sample['encoder_target'])
                ground_truth_arch_seq_list.extend(sample['decoder_target'])

        pairwise_acc = nao_utils.pairwise_accuracy(ground_truth_perf_list, predict_value_list)
        hamming_dis = nao_utils.hamming_distance(ground_truth_arch_seq_list, arch_seq_list)

        if pairwise_acc > self._ctrl_best_pa[subset]:
            self._ctrl_best_pa[subset] = pairwise_acc

        time.stop()
        logging.info('| ctrl eval ({}) | epoch {:03d} | PA {:<6.6f} | BestPA {:<6.6f} |'
                     ' HD {:<6.6f} | {:<6.2f} secs'.format(
                      subset, epoch, pairwise_acc, self._ctrl_best_pa[subset], hamming_dis, time.sum))

    def controller_generate_step(self, old_arches, log_compare_perf=False):
        epd = self.controller.epd

        old_arches = old_arches[:self.hparams.num_remain_top]

        # print('#old_arches:', [a.blocks for a in old_arches])

        new_arches = []
        mapped_old_perf_list = []
        final_old_perf_list, final_new_perf_list = [], []

        predict_lambda = 0
        topk_arches = [self._parse_arch_to_seq(arch) for arch in old_arches[:self.hparams.num_pred_top]]
        topk_arches_loader = nao_utils.make_tensor_dataloader(
            [th.LongTensor(topk_arches)], self.hparams.ctrl_batch_size, shuffle=False)

        with th.cuda.device(self.hparams.epd_device):
            while len(new_arches) + len(old_arches) < self.hparams.num_seed_arch:
                # [NOTE]: When predict_lambda get larger, increase faster.
                if predict_lambda < 50:
                    predict_lambda += self.hparams.lambda_step
                elif predict_lambda >= 10000000:
                    # FIXME: A temporary solution: stop the generation when the lambda is too large.
                    break
                else:
                    predict_lambda += predict_lambda / 50
                logging.info('Generating new architectures using gradient descent with step size {}'.format(
                    predict_lambda))

                new_arch_seq_list = []
                for step, (encoder_input,) in enumerate(topk_arches_loader):
                    epd.eval()
                    epd.zero_grad()
                    encoder_input = common.make_variable(encoder_input, volatile=False, cuda=True)
                    new_arch_seq, ret_dict = epd.generate_new_arch(encoder_input, predict_lambda)
                    new_arch_seq_list.extend(new_arch_seq.data.squeeze().tolist())
                    mapped_old_perf_list.extend(ret_dict['predict_value'].squeeze().tolist())
                    del ret_dict

                for i, (arch_seq, mapped_perf) in enumerate(zip(new_arch_seq_list, mapped_old_perf_list)):
                    # Insert new arches (skip same and invalid).
                    # [NOTE]: Reduce the "ctrl_trade_off" value to let it generate different architectures.
                    arch = self.controller.parse_seq_to_arch(arch_seq)
                    if arch is None:
                        continue

                    if not self._arch_contains(arch, old_arches) and not self._arch_contains(arch, new_arches):
                        new_arches.append(arch)
                        # Test the new arch.
                        sample = nao_utils.prepare_ctrl_sample(
                            [th.LongTensor([arch_seq]), th.LongTensor([arch_seq]), th.LongTensor([arch_seq])],
                            perf=False,
                        )
                        predict_value, _, _ = epd(sample['encoder_input'], sample['decoder_input'])
                        final_old_perf_list.append(mapped_perf)
                        final_new_perf_list.append(predict_value.item())
                    if len(new_arches) + len(old_arches) >= self.hparams.num_seed_arch:
                        break
                logging.info('{} new arches generated now'.format(len(new_arches)))

        # Compare old and new perf.
        if log_compare_perf:
            print('Old and new performances:')
            _s_old, _s_new = 0.0, 0.0
            for _old, _new in zip(final_old_perf_list, final_new_perf_list):
                print('old = {}, new = {}, old - new = {}'.format(_old, _new, _old - _new))
                _s_old += _old
                _s_new += _new
            _s_old /= len(final_new_perf_list)
            _s_new /= len(final_new_perf_list)
            print('Average: old = {}, new = {}, old - new = {}'.format(_s_old, _s_new, _s_old - _s_new))

        self.arch_pool = old_arches + new_arches
        return self.arch_pool

    def save_epd(self):
        save_path = get_model_path(self.hparams)
        os.makedirs(save_path, exist_ok=True)
        filename = 'epd_checkpoint{}.pt'.format(self.hparams.sa_iteration)
        state = {
            'sa_iteration': self.hparams.sa_iteration,
            'epd': self.controller.epd.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        th.save(state, os.path.join(save_path, filename))
        logging.info('Save checkpoint to {}'.format(filename))

    def load_epd(self):
        filename = os.path.join(get_model_path(self.hparams), 'epd_checkpoint{}.pt'.format(self.hparams.sa_iteration))
        state = th.load(filename)
        assert state['sa_iteration'] == self.hparams.sa_iteration
        self.controller.epd.load_state_dict(state['epd'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])
        logging.info('Load checkpoint from {}'.format(filename))

    @staticmethod
    def _arch_contains(arch, arch_list):
        return any(arch.fast_eq(a) for a in arch_list)


def nao_epd_main(hparams):
    # TODO: Add stand alone training script of NaoEpd.
    import json
    from libs.layers.net_code import NetCode

    DirName = 'D:/Users/v-yaf/DataTransfer/NAS4Text/arch_pool_results'
    iteration = hparams.sa_iteration
    subset = 'dev'

    components = mu.main_entry(
        hparams, train=True, net_code='nao_train_standalone', hparams_ppp=nao_utils.hparams_ppp_nao)
    datasets = components['datasets']

    logging.info('Building model')
    criterion = build_criterion(hparams, datasets.source_dict, datasets.target_dict)
    trainer = NAOTrainer(hparams, criterion, only_epd_cuda=True)
    model = trainer.get_model()
    mu.logging_model_criterion(model, criterion, logging_params=False)
    mu.logging_training_stats(hparams)

    TargetFiles = {
        'x': os.path.join(DirName, 'arches-{}-{}-{}.txt'.format(hparams.hparams_set, subset, iteration)),
        'y': os.path.join(DirName, 'bleus-{}-{}-{}.txt'.format(hparams.hparams_set, subset, iteration)),
        'output': os.path.join(DirName, 'arches-{}-{}-{}.txt'.format(hparams.hparams_set, subset, iteration)),
    }

    with open(TargetFiles['x'], 'r', encoding='utf-8') as f_x, \
            open(TargetFiles['y'], 'r', encoding='utf-8') as f_y:
        arch_pool = [NetCode(json.loads(line)) for line in f_x]
        perf_pool = [1.0 - float(line.strip()) / 100.0 for line in f_y]

    # Sort old arches.
    arches_sorted_indices = np.argsort(perf_pool)
    arch_pool = [arch_pool[i] for i in arches_sorted_indices]
    perf_pool = [perf_pool[i] for i in arches_sorted_indices]

    trainer.arch_pool = arch_pool

    if hparams.reload:
        trainer.load_epd()
    else:
        split_test = True
        trainer.controller_train_step(
            arch_pool, perf_pool, split_test=split_test)

    new_arch_pool = trainer.controller_generate_step(arch_pool, log_compare_perf=True)
    # [NOTE]: Only save unique arches.
    unique_arch_pool = []
    for arch in new_arch_pool:
        if not any(arch.fast_eq(a) for a in arch_pool):
            unique_arch_pool.append(arch)
    nao_utils.save_arches(hparams, iteration, unique_arch_pool, arches_perf=None, after_gen=True)

    trainer.save_epd()


def nao_search_main(hparams):
    components = mu.main_entry(
        hparams, train=True, net_code='nao_train', hparams_ppp=nao_utils.hparams_ppp_nao)
    datasets = components['datasets']

    logging.info('Building model')
    criterion = build_criterion(hparams, datasets.source_dict, datasets.target_dict)
    trainer = NAOTrainer(hparams, criterion)
    model = trainer.get_model()
    mu.logging_model_criterion(model, criterion, logging_params=False)
    mu.logging_training_stats(hparams)

    # Used to skip child training and evaluation in debug mode.
    debug_epd = False

    max_ctrl_step = hparams.max_ctrl_step or math.inf
    ctrl_step = 1
    train_meter = StopwatchMeter()
    train_meter.start()
    while ctrl_step <= max_ctrl_step:
        logging.info('Training step {}'.format(ctrl_step))
        trainer.set_seed(ctrl_step)

        # Train child model.
        trainer.init_arch_pool()

        if debug_epd:
            valid_acc_list = list(np.linspace(0.0, 1.0, len(trainer.arch_pool)))
        else:
            trainer.train_children(datasets)

            # Evaluate seed arches.
            valid_acc_list = trainer.eval_children(datasets, compute_loss=False)

        # Output arches and evaluated error rate.
        old_arches = trainer.arch_pool
        # Error rate list.
        old_arches_perf = [1.0 - i for i in valid_acc_list]

        # Sort old arches.
        old_arches_sorted_indices = np.argsort(old_arches_perf)
        old_arches = [old_arches[i] for i in old_arches_sorted_indices]
        old_arches_perf = [old_arches_perf[i] for i in old_arches_sorted_indices]

        # Save old arches and performances in order.
        nao_utils.save_arches(hparams, ctrl_step, old_arches, old_arches_perf)

        # Train encoder-predictor-decoder.
        trainer.controller_train_step(old_arches, old_arches_perf)

        # Generate new arches.
        trainer.controller_generate_step(old_arches)

        # Save updated arches after generate step.
        nao_utils.save_arches(hparams, ctrl_step, trainer.arch_pool, arches_perf=None, after_gen=True)

        # TODO: Save shared weights.

        ctrl_step += 1
    train_meter.stop()
    logging.info('Training done in {:.1f} seconds'.format(train_meter.sum))


def main(args=None):
    hparams = nao_utils.get_nao_search_args(args)
    if hparams.standalone:
        nao_epd_main(hparams)
    else:
        nao_search_main(hparams)


if __name__ == '__main__':
    main()
