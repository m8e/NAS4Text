#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Classes and functions for data processing.

Language dataset format:

```
# Assume source language = de, target language = en
dataset-name/
    train.de-en.de
    train.de-en.en
    dev.de-en.de
    dev.de-en.en
    test.de-en.de
    test.de-en.en
    dict.de-en.de
    dict.de-en.en
```

Dict format: pickled dict
    Key: token string
    Value: token id
    Notes:
        Contains 3 special tokens: padding '<pad>' = 0, eos '</s>' = 1, unknown '<unk>' = 2.

"""

import os
import pickle

from torch.utils.data import Dataset

from .paths import DataDir
from ..tasks import get_task

__author__ = 'fyabc'


class LanguageDatasets:
    """Container of all dataset splits of the task."""
    def __init__(self, task_name):
        self.task = get_task(task_name)
        self.splits = {}

        dataset_dir = os.path.join(DataDir, self.task.TaskName)

        # Load dictionary.
        with open(os.path.join(dataset_dir, self.task.SourceFiles['dict']), 'rb') as f:
            self.source_dict = pickle.load(f, encoding='utf-8')
        with open(os.path.join(dataset_dir, self.task.TargetFiles['dict']), 'rb') as f:
            self.target_dict = pickle.load(f, encoding='utf-8')

        self._check_dict()

    def _check_dict(self):
        assert len(self.source_dict) == self.task.SourceVocabSize, 'Incorrect source vocabulary size'
        assert len(self.target_dict) == self.task.TargetVocabSize, 'Incorrect target vocabulary size'

        assert self.source_dict.get(self.task.PAD, None) == self.task.PAD_ID, 'Incorrect source PAD id'
        assert self.target_dict.get(self.task.PAD, None) == self.task.PAD_ID, 'Incorrect target PAD id'
        assert self.source_dict.get(self.task.EOS, None) == self.task.EOS_ID, 'Incorrect source EOS id'
        assert self.target_dict.get(self.task.EOS, None) == self.task.EOS_ID, 'Incorrect target EOS id'
        assert self.source_dict.get(self.task.UNK, None) == self.task.UNK_ID, 'Incorrect source UNK id'
        assert self.target_dict.get(self.task.UNK, None) == self.task.UNK_ID, 'Incorrect target UNK id'


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
