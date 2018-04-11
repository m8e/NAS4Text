#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import pickle

from torch.utils.data import Dataset

from .paths import DataDir
from ..tasks import get_task

__author__ = 'fyabc'


class LanguageDatasets:
    def __init__(self, task_name):
        self.task = get_task(task_name)
        self.splits = {}

        dataset_dir = os.path.join(DataDir, self.task.TaskName)

        with open(os.path.join(dataset_dir, self.task.SourceFiles['dict']), 'rb') as f:
            self.source_dict = pickle.load(f, encoding='utf-8')
        with open(os.path.join(dataset_dir, self.task.TargetFiles['dict']), 'rb') as f:
            self.target_dict = pickle.load(f, encoding='utf-8')

        assert len(self.source_dict) == self.task.SourceVocabSize, 'Incorrect source vocabulary size'
        assert len(self.target_dict) == self.task.TargetVocabSize, 'Incorrect target vocabulary size'


class LanguagePairDataset(Dataset):
    # Padding constants
    LEFT_PAD_SOURCE = True
    LEFT_PAD_TARGET = False

    def __init__(self, src, trg, pad_idx, eos_idx):
        self.src = src
        self.trg = trg
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
