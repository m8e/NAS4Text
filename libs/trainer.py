#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from .child_net import ChildNet
from .utils.data_processing import LanguageDatasets, LanguagePairDataset
from .utils.common import mask_from_lengths

__author__ = 'fyabc'


class Trainer:
    def __init__(self, hparams, net_code):
        self.hparams = hparams
        self.model = ChildNet(net_code, hparams)
        self.datasets = LanguageDatasets(hparams.task)

        # TODO: Replace them in future
        self.criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=self.task.PAD_ID)
        self.optimizer = optim.Adadelta(self.model.parameters())

    @property
    def task(self):
        return self.datasets.task

    @property
    def task_name(self):
        return self.datasets.task.TaskName

    def train(self):
        # TODO: Apply hparams of all args of this (see fairseq-py for details)
        train_dataloader = self.datasets.train_dataloader(
            'train', max_tokens=self.hparams.max_tokens,
            max_sentences=self.hparams.max_sentences,
            max_positions=(self.hparams.max_src_positions, self.hparams.max_trg_positions),
            seed=None, epoch=1, sample_without_replacement=0,
            sort_by_source_size=False, shard_id=0, num_shards=1)
        for epoch in range(self.hparams.max_epoch):
            for batch in train_dataloader:
                pred_trg_probs = self.model(
                    src_tokens=Variable(batch['net_input']['src_tokens']),
                    src_lengths=Variable(batch['net_input']['src_lengths']),
                    trg_tokens=Variable(batch['net_input']['trg_tokens']),
                    trg_lengths=Variable(batch['net_input']['trg_lengths']),
                )
                self.optimizer.zero_grad()

                target = Variable(batch['target'])
                loss = self.criterion(
                    pred_trg_probs.view(-1, pred_trg_probs.size(-1)),
                    target.view(-1),
                )
                loss.backward()

                print('Loss = {}'.format(loss.data[0]))

                self.optimizer.step()

                corrects = target == pred_trg_probs.max(dim=-1)[1]
                mask = mask_from_lengths(batch['net_input']['trg_lengths'], LanguagePairDataset.LEFT_PAD_TARGET)
                corrects = th.masked_select(corrects.data, mask)
                print('Argmax error rate:', (1.0 - corrects.float().sum() / corrects.nelement()))
