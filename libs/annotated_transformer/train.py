#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import math
import os
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from .utils import fix_batch
from .model import make_model
from ..utils.main_utils import main_entry
from ..utils.common import make_variable, torch_persistent_save
from ..utils.paths import get_model_path
from ..utils.meters import StopwatchMeter

__author__ = 'fyabc'


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class SimpleTrainer:
    def __init__(self, hparams, model, criterion, optimizer, datasets):
        self.hparams = hparams
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.datasets = datasets

        self._max_bsz_seen = 0

    def _get_train_iter(self, seed, epoch):
        hparams = self.hparams
        datasets = self.datasets
        max_positions_train = hparams.max_src_positions - datasets.task.PAD_ID - 1

        return datasets.train_dataloader(
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

    def _prepare_sample(self, sample, volatile):
        if sample is None or len(sample) == 0:
            return None
        if hasattr(torch.cuda, 'empty_cache'):
            # Clear the caching allocator if this is the largest sample we've seen
            if sample['target'].size(0) > self._max_bsz_seen:
                self._max_bsz_seen = sample['target'].size(0)
                torch.cuda.empty_cache()

        return make_variable(sample, volatile=volatile, cuda=True)

    def _loss_compute(self, model_output, y_trg, norm):
        y_pred = self.model.generator(model_output)
        loss = self.criterion(y_pred.contiguous().view(-1, y_pred.size(-1)),
                              y_trg.contiguous().view(-1)) / norm
        loss.backward()
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.optimizer.zero_grad()
        return loss.data[0] * norm

    def train(self, epoch, batch_offset):
        # Set seed based on args.seed and the epoch number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.hparams.seed + epoch
        torch.manual_seed(seed)

        train_iter = self._get_train_iter(seed, epoch)
        train_len = len(train_iter)

        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0

        for i, batch in enumerate(train_iter, start=1):
            fix_batch(batch, pad_id=self.datasets.task.PAD_ID)
            sample = self._prepare_sample(batch, volatile=False)

            self.model.train()

            net_input = sample['net_input']
            model_output = self.model(net_input['src_tokens'], net_input['trg_tokens'],
                                      net_input['src_mask'], net_input['trg_mask'])
            loss = self._loss_compute(model_output, sample['target'], sample['ntokens'])
            total_loss += loss
            total_tokens += sample['ntokens']
            tokens += sample['ntokens']

            if i % self.hparams.log_interval == 0:
                elapsed = time.time() - start
                logging.info("Epoch Step: {}/{} Loss: {:.6f} Tokens per Sec: {:.6f}".format(
                    i, train_len, loss / sample['ntokens'], tokens / elapsed))
                start = time.time()
                tokens = 0

        return total_loss / total_tokens

    def save_checkpoint(self, epoch):
        save_dir = get_model_path(self.hparams)
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {
            'hparams': self.hparams,
            'model': self.model.state_dict(),
            'extra_state': {
                'epoch': epoch,
                'batch_offset': 0,
            }
        }
        epoch_filename = os.path.join(save_dir, 'checkpoint{}.pt'.format(epoch))
        torch_persistent_save(state_dict, epoch_filename)
        logging.info('Save checkpoint to {} (epoch {})'.format(epoch_filename, epoch))


def annotated_train_main(hparams):
    components = main_entry(hparams, train=True, net_code=False)

    hparams.net_code_file = 'annotated_transformer'

    datasets = components['datasets']

    # FIXME: Hard-code hparams here.
    model = make_model(
        hparams=hparams,
        src_vocab=datasets.task.get_vocab_size(is_src_lang=True),
        tgt_vocab=datasets.task.get_vocab_size(is_src_lang=False),
        N=2,
        d_model=256,
        d_ff=1024,
        h=8,
        dropout=0.1,
    )
    model.cuda()

    criterion = LabelSmoothing(
        size=datasets.task.get_vocab_size(is_src_lang=False),
        padding_idx=datasets.task.PAD_ID,
        smoothing=0.1,
    )
    criterion.cuda()

    optimizer = NoamOpt(model.d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(.9, .98), eps=1e-9))

    trainer = SimpleTrainer(hparams, model, criterion, optimizer, datasets)

    max_epoch = hparams.max_epoch or math.inf
    epoch, batch_offset = 1, 0

    train_meter = StopwatchMeter()
    train_meter.start()
    while epoch < max_epoch:
        epoch_loss = trainer.train(epoch, batch_offset=batch_offset)
        logging.info('Epoch: {} Loss: {:.6f}'.format(epoch, epoch_loss))
        trainer.save_checkpoint(epoch)

        epoch += 1
        batch_offset = 0
    train_meter.stop()
    logging.info('Training done in {:.1f} seconds'.format(train_meter.sum))
