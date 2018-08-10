#! /usr/bin/python
# -*- coding: utf-8 -*-

# TODO: Support multi-card parallel training.

import collections
import itertools
import logging
import math
import os

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from libs.models.darts_child_net import DartsChildNet, ParalleledDartsChildNet
from libs.optimizers import build_optimizer
from libs.criterions import build_criterion
from libs.hparams import hparams_env
from libs.utils.args import get_darts_search_args
from libs.utils import main_utils as mu
from libs.utils.meters import StopwatchMeter, AverageMeter
from libs.utils.progress_bar import build_progress_bar
from libs.utils.paths import get_model_path
from libs.child_trainer import ChildTrainer

__author__ = 'fyabc'


th_grad = autograd.grad


UseParallel = False     # [NOTE]: A debug flag.


def _concat(xs):
    return th.cat([x.view(-1) for x in xs])


class DartsTrainer(ChildTrainer):
    def __init__(self, hparams, model, criterion):
        super().__init__(hparams, model, criterion)

        # [NOTE]: In DARTS, optimizer is fixed to Momentum SGD, and lr scheduler is fixed to CosineAnnealingLR.
        assert hparams.optimizer == 'sgd', 'DARTS training must use SGD as optimizer'

        # [NOTE]: In DARTS, arch optimizer is fixed to adam, and no arch lr scheduler.
        with hparams_env(
            hparams,
            lr=[hparams.arch_lr], adam_betas=hparams.arch_adam_betas, adam_eps=1e-8,
            weight_decay=hparams.arch_weight_decay,
        ) as arch_hparams:
            self.arch_optimizer = build_optimizer(arch_hparams, self.model.arch_parameters())
            logging.info('Arch optimizer: {}'.format(self.arch_optimizer.__class__.__name__))

        self.network_momentum = hparams.momentum
        self.network_weight_decay = hparams.weight_decay

    def search_step(self, sample, search_sample):
        self.model.train()
        self.arch_optimizer.zero_grad()

        # Sometimes train batches and search batches are not aligned, just ignore them.
        if search_sample is None:
            return 0.0

        if self.hparams.unrolled:
            self._backward_step_unrolled(sample, search_sample)
        else:
            self._backward_step(search_sample)

        grad_norm = nn.utils.clip_grad_norm_(self.model.arch_parameters(), self.hparams.arch_clip_norm)
        # FIXME: Add number of updates or not?
        self.arch_optimizer.step()
        return grad_norm.item()

    def _backward_step_unrolled(self, sample, search_sample):
        sample = self._prepare_sample(sample, volatile=False)
        search_sample = self._prepare_sample(search_sample, volatile=False)

        model_unrolled = self._compute_unrolled_model(sample)
        loss, sample_size, logging_output_ = self.criterion(model_unrolled, search_sample)
        grads = th_grad(loss, model_unrolled.arch_parameters(), retain_graph=True)
        
        theta = model_unrolled.parameters()
        d_theta = th_grad(loss, model_unrolled.parameters())
        vector = [dt.add(self.network_weight_decay, t).data for dt, t in zip(d_theta, theta)]
        implicit_grads = self._hessian_vector_product(model_unrolled, vector, sample)

        for g, ig in zip(grads, implicit_grads):
            g.data.sub_(self.get_lr(), ig.data)

        for v, g in zip(self.model.arch_parameters(), grads):
            if v.grad is None:
                v.grad = autograd.Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _backward_step(self, search_sample):
        search_sample = self._prepare_sample(search_sample, volatile=False)
        loss, sample_size, logging_output_ = self.criterion(self.model, search_sample)
        for v in self.model.arch_parameters():
            if v.grad is not None:
                v.grad.data.zero_()
        loss.backward()

    def _compute_unrolled_model(self, sample):
        loss, sample_size, logging_output_ = self.criterion(self.model, sample)
        theta = _concat(self.model.parameters()).data
        try:
            state = self.optimizer.optimizer.state
            moment = _concat(state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
        except KeyError:
            moment = th.zeros_like(theta)
        d_theta = _concat(th_grad(loss, self.model.parameters())).data + \
            self.network_weight_decay * theta
        return self._construct_model_from_theta(theta.sub(self.get_lr(), moment + d_theta))
    
    def _construct_model_from_theta(self, theta):
        model_clone = DartsChildNet(self.hparams)

        # TODO: Fix the problem of grad computation and updates in parallel training.

        if isinstance(self.model, ParalleledDartsChildNet):
            model = self.model.module
        else:
            model = self.model

        for x, y in zip(model_clone.arch_parameters(), model.arch_parameters()):
            x.data.copy_(y.data)
        model_dict = model.state_dict()

        params, offset = {}, 0
        for k, v in model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_clone.load_state_dict(model_dict)
        
        return model_clone.cuda()
    
    def _hessian_vector_product(self, model, vector, sample, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)
        loss, _, _ = self.criterion(model, sample)
        grads_p = th_grad(loss, model.arch_parameters())
        
        for p, v in zip(model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss, _, _ = self.criterion(model, sample)
        grads_n = th_grad(loss, model.arch_parameters())

        for p, v in zip(model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


def darts_search_main(hparams):
    components = mu.main_entry(hparams, train=True, net_code='darts')
    datasets = components['datasets']

    logging.info('Building model')
    model = DartsChildNet(hparams)
    # TODO: Add ParalleledChildNet here.
    if UseParallel:
        model = ParalleledDartsChildNet(model, output_device=hparams.device_id)

    # [NOTE]: In DARTS, criterion is fixed to CrossEntropyLoss.
    criterion = build_criterion(hparams, datasets.source_dict, datasets.target_dict)
    mu.logging_model_criterion(model, criterion)

    trainer = DartsTrainer(hparams, model, criterion)
    mu.logging_training_stats(hparams)

    epoch, batch_offset = mu.prepare_checkpoint(hparams, trainer)

    # Send a dummy batch to warm the caching allocator
    dummy_batch = datasets.get_dataset('train').get_dummy_batch(hparams.max_tokens, trainer.get_model().max_positions())
    trainer.dummy_train_step(dummy_batch)

    # Train until the learning rate gets too small
    max_epoch = hparams.max_epoch or math.inf
    max_update = hparams.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > hparams.min_lr and epoch <= max_epoch:
        # Train for one epoch
        train(hparams, trainer, datasets, epoch, batch_offset)

        # Evaluate on validate set
        if epoch % hparams.validate_interval == 0:
            for k, subset in enumerate(hparams.valid_subset.split(',')):
                # TODO: Replace with my validate
                val_loss = mu.validate(hparams, trainer, datasets, subset, epoch)
                if k == 0:
                    # Only use first validation loss to update the learning schedule
                    lr = trainer.lr_step(epoch, val_loss)

                    # save checkpoint and net code
                    if not hparams.no_save:
                        # TODO: Replace with my save_checkpoint (or not?)
                        mu.save_checkpoint(trainer, hparams, epoch, 0, val_loss)
                        save_net_code(trainer, hparams, epoch, 0, val_loss)
        else:
            lr = trainer.lr_step(epoch)

        epoch += 1
        batch_offset = 0

        if trainer.get_num_updates() >= max_update:
            break
    train_meter.stop()
    logging.info('Training done in {:.1f} seconds'.format(train_meter.sum))


def train(hparams, trainer, datasets, epoch, batch_offset):
    """Train the model for one epoch.

    Args:
        hparams:
        trainer (DartsTrainer):
        datasets (LanguageDatasets):
        epoch (int):
        batch_offset (int):

    Returns:

    """

    # Set seed based on args.seed and the epoch number so that we get
    # reproducible results when resuming from checkpoints
    seed = hparams.seed
    th.manual_seed(seed)

    # The max number of positions can be different for train and valid
    # e.g., RNNs may support more positions at test time than seen in training
    max_positions_train = (
        min(hparams.max_src_positions, trainer.get_model().max_encoder_positions()),
        min(hparams.max_trg_positions, trainer.get_model().max_decoder_positions()),
    )

    # Initialize dataloader, starting at batch_offset
    # TODO: Split the training data into half-train and half search.
    train_search_split = int(math.floor(len(datasets.get_dataset(hparams.train_subset)) * hparams.train_portion))
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
        start=0,
        end=train_search_split,
    )

    # Initialize search dataloader from training data.
    search_itr = datasets.train_dataloader(
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
        start=train_search_split,
        end=None,
    )

    next(itertools.islice(itr, batch_offset, batch_offset), None)
    progress = build_progress_bar(hparams, itr, epoch, no_progress_bar='simple')

    # Reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    max_update = hparams.max_update or math.inf
    for i, sample in enumerate(progress, start=batch_offset):
        search_sample = next(iter(search_itr))

        arch_grad_norm = trainer.search_step(sample, search_sample)
        extra_meters['arch_gnorm'].update(arch_grad_norm)

        log_output = trainer.train_step(sample)

        # Log mid-epoch stats
        stats = mu.get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        stats['arch_gnorm'] = '{:.3f}'.format(extra_meters['arch_gnorm'].avg)
        progress.log(stats)

        if i == batch_offset:
            # Ignore the first mini-batch in words-per-second calculation
            trainer.get_meter('wps').reset()

        # save mid-epoch checkpoints
        num_updates = trainer.get_num_updates()
        if hparams.save_interval > 0 and num_updates > 0 and num_updates % hparams.save_interval == 0:
            mu.save_checkpoint(trainer, hparams, epoch, i + 1)
            save_net_code(trainer, hparams, epoch, i + 1)

        if num_updates >= max_update:
            break

    # Log end-of-epoch stats
    stats = mu.get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    print('Alphas after epoch {}:\n{}'.format(
        epoch, '\n'.join(
            str(F.softmax(a, dim=-1)) for a in trainer.get_model().arch_parameters())
        ),
    )


def save_net_code(trainer, hparams, epoch, batch_offset, val_loss=None):
    import json

    save_dir = get_model_path(hparams)
    net_code = trainer.get_model().dump_net_code(branch=2)

    if batch_offset == 0:
        if not hparams.no_epoch_checkpoints:
            epoch_filename = os.path.join(save_dir, 'net_code{}.json'.format(epoch))
            with open(epoch_filename, 'w', encoding='utf-8') as f:
                json.dump(net_code, f, indent=4)
            logging.info('Save net code to {} (epoch {})'.format(epoch_filename, epoch))

        assert val_loss is not None
        if not hasattr(save_net_code, 'best') or val_loss < save_net_code.best:
            save_net_code.best = val_loss
            best_filename = os.path.join(save_dir, 'net_code_best.json')
            with open(best_filename, 'w', encoding='utf-8') as f:
                json.dump(net_code, f, indent=4)
            logging.info('Save net code to {} (epoch {})'.format(best_filename, epoch))
    elif not hparams.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            save_dir, 'net_code{}_{}.json'.format(epoch, batch_offset))
        with open(epoch_filename, 'w', encoding='utf-8') as f:
            json.dump(net_code, f, indent=4)
        logging.info('Save net code to {} (epoch {})'.format(epoch_filename, epoch))

    last_filename = os.path.join(save_dir, 'net_code_last.json')
    with open(last_filename, 'w', encoding='utf-8') as f:
        json.dump(net_code, f, indent=4)
    logging.info('Save net code to {} (epoch {})'.format(last_filename, epoch))


def main(args=None):
    hparams = get_darts_search_args(args)
    darts_search_main(hparams)


if __name__ == '__main__':
    main()
