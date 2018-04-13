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

import numpy as np
from torch.utils.data import Dataset, DataLoader

from .paths import DataDir
from .dictionary import Dictionary
from ..tasks import get_task
from .tokenizer import Tokenizer

__author__ = 'fyabc'


class LanguageDatasets:
    """Container of all dataset splits of the task."""
    def __init__(self, task_name, split_names):
        self.task = get_task(task_name)
        self.splits = {}

        self.dataset_dir = os.path.join(DataDir, self.task.TaskName)

        # Load dictionary.
        with open(os.path.join(self.dataset_dir, self.task.get_filename('dict', is_src_lang=True)), 'rb') as f:
            self.source_dict = Dictionary(pickle.load(f, encoding='utf-8'), self.task)
        with open(os.path.join(self.dataset_dir, self.task.get_filename('dict', is_src_lang=False)), 'rb') as f:
            self.target_dict = Dictionary(pickle.load(f, encoding='utf-8'), self.task)

    def train_dataloader(self, split, max_tokens=None,
                         max_sentences=None, max_positions=(1024, 1024),
                         seed=None, epoch=1, sample_without_replacement=0,
                         sort_by_source_size=False, shard_id=0, num_shards=1):
        dataset = self._get_dataset(split)

        # TODO

    def eval_dataloader(self, split, num_workers=0, max_tokens=None,
                        max_sentences=None, max_positions=(1024, 1024),
                        skip_invalid_size_inputs_valid_test=False,
                        descending=False, shard_id=0, num_shards=1):
        dataset = self._get_dataset(split)

        # TODO

    def _get_dataset(self, split_name):
        if split_name not in self.splits:
            src_path = self.task.get_filename(split_name, is_src_lang=True)
            trg_path = self.task.get_filename(split_name, is_src_lang=False)
            self.splits[split_name] = LanguagePairDataset(
                TextDataset(src_path, self.source_dict),
                TextDataset(trg_path, self.target_dict),
                pad_id=self.source_dict.pad_id,
                eos_id=self.source_dict.eos_id,
            )

        return self.splits[split_name]


class TextDataset:
    def __init__(self, path, dictionary, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        self.check_index(index)
        return self.tokens_list[index]

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = Tokenizer.tokenize(line, dictionary, add_if_not_exist=False,
                                            append_eos=self.append_eos, reverse_order=self.reverse_order)
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def get_original_text(self, index):
        self.check_index(index)
        return self.lines[index]


class LanguagePairDataset(Dataset):
    """Language pair dataset.

    Contains two `TextDataset` of source and target language.
    """

    # Padding constants
    LEFT_PAD_SOURCE = True
    LEFT_PAD_TARGET = False

    def __init__(self, src, trg, pad_id, eos_id):
        self.src = src
        self.trg = trg
        self.pad_id = pad_id
        self.eos_id = eos_id

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass

    def collater(self, samples):
        return self.collate(samples, self.pad_id, self.eos_id, self.trg is not None)

    @staticmethod
    def collate(samples, pad_id, eos_id, has_target=True):
        # TODO

        pass
