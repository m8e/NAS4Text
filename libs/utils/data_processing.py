#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Classes and functions for data processing.

Mainly copied from fairseq-py.

Language dataset format:

```
# Assume source language = de, target language = en, unique name = iwslt
dataset-name/
    train.iwslt.de-en.de
    train.iwslt.de-en.en
    dev.iwslt.de-en.de
    dev.iwslt.de-en.en
    test.iwslt.de-en.de
    test.iwslt.de-en.en
    dict.iwslt.de-en.de
    dict.iwslt.de-en.en
```

Dict format: pickled dict
    Key: token string
    Value: token id
    Notes:
        Contains 3 special tokens: padding '<pad>' = 0, eos '</s>' = 1, unknown '<unk>' = 2.

"""

import math
import os
import pickle
import numbers

import numpy as np
import torch as th
from torch.utils.data import Dataset, DataLoader

from .paths import DataDir
from .dictionary import Dictionary
from ..tasks import get_task
from .tokenizer import Tokenizer
from .common import numpy_seed

__author__ = 'fyabc'


class LanguageDatasets:
    """Container of all dataset splits of the task."""
    def __init__(self, task_name):
        self.task = get_task(task_name)
        self.splits = {}

        self.dataset_dir = os.path.join(DataDir, self.task.TaskName)

        # Load dictionary.
        with open(os.path.join(self.dataset_dir, self.task.get_filename('dict', is_src_lang=True)), 'rb') as f:
            self.source_dict = Dictionary(pickle.load(f, encoding='utf-8'), self.task, is_src_lang=True)
        with open(os.path.join(self.dataset_dir, self.task.get_filename('dict', is_src_lang=False)), 'rb') as f:
            self.target_dict = Dictionary(pickle.load(f, encoding='utf-8'), self.task, is_src_lang=False)

    def train_dataloader(self, split, max_tokens=None,
                         max_sentences=None, max_positions=(1024, 1024),
                         seed=None, epoch=1, sample_without_replacement=0,
                         sort_by_source_size=False, shard_id=0, num_shards=1):
        dataset = self.get_dataset(split)

        with numpy_seed(seed):
            batch_sampler = dataset.shuffled_batches_by_size(
                max_tokens=max_tokens,
                max_sentences=max_sentences, epoch=epoch,
                sample=sample_without_replacement, max_positions=max_positions,
                sort_by_source_size=sort_by_source_size)
            batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)

        return DataLoader(
            dataset, collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
        )

    def eval_dataloader(self, split, num_workers=0, max_tokens=None,
                        max_sentences=None, max_positions=(1024, 1024),
                        skip_invalid_size_inputs_valid_test=False,
                        descending=False, shard_id=0, num_shards=1):
        dataset = self.get_dataset(split)

        batch_sampler = dataset.batches_by_size(
            max_tokens=max_tokens, max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs_valid_test,
            descending=descending)
        batch_sampler = mask_batches(batch_sampler, shard_id=shard_id, num_shards=num_shards)

        return DataLoader(
            dataset, num_workers=num_workers,
            collate_fn=dataset.collater, batch_sampler=batch_sampler,
        )

    def get_dataset(self, split_name):
        """Get the language pair dataset. Load it if not loaded.

        Args:
            split_name: Name of the split to load.

        Returns:
            LanguagePairDataset
        """
        if split_name not in self.splits:
            src_path = os.path.join(self.dataset_dir, self.task.get_filename(split_name, is_src_lang=True))
            trg_path = os.path.join(self.dataset_dir, self.task.get_filename(split_name, is_src_lang=False))
            self.splits[split_name] = LanguagePairDataset(
                TextDataset(src_path, self.source_dict),
                TextDataset(trg_path, self.target_dict),
                pad_id=self.source_dict.pad_id,
                eos_id=self.source_dict.eos_id,
            )

        return self.splits[split_name]

    def load_splits(self, splits):
        for split in splits:
            self.get_dataset(split)


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
    # [NOTE]: It seems that this constant MUST be True for both source and target,
    # because
    LEFT_PAD_SOURCE = False     # True in fairseq-py
    LEFT_PAD_TARGET = False

    def __init__(self, src, trg, pad_id, eos_id):
        self.src = src
        self.trg = trg
        self.pad_id = pad_id
        self.eos_id = eos_id

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        source = self.src[index]
        result = {
            'id': index,
            'source': source,
        }
        if self.trg:
            result['target'] = self.trg[index]
        return result

    def batches_by_size(self, max_tokens=None, max_sentences=None,
                        max_positions=(1024, 1024), ignore_invalid_inputs=False,
                        descending=False):
        """Returns batches of indices sorted by size. Sequences with different
        source lengths are not allowed in the same batch.
        """
        if max_tokens is None:
            max_tokens = float('Inf')
        if max_sentences is None:
            max_sentences = float('Inf')
        indices = np.argsort(self.src.sizes, kind='mergesort')
        if descending:
            indices = np.flip(indices, 0)
        return list(_make_batches(
            self.src, self.trg, indices, max_tokens, max_sentences, max_positions,
            ignore_invalid_inputs, allow_different_src_lens=False))

    def shuffled_batches_by_size(self, max_tokens=None, max_sentences=None,
                                 epoch=1, sample=0, max_positions=(1024, 1024),
                                 sort_by_source_size=False):
        """Returns batches of indices, bucketed by size and then shuffled. Batches
        may contain sequences of different lengths."""
        if max_tokens is None:
            max_tokens = float('Inf')
        if max_sentences is None:
            max_sentences = float('Inf')

        indices = np.random.permutation(len(self.src))

        # sort by sizes
        indices = indices[np.argsort(self.trg.sizes[indices], kind='mergesort')]
        indices = indices[np.argsort(self.src.sizes[indices], kind='mergesort')]

        batches = list(_make_batches(
            self.src, self.trg, indices, max_tokens, max_sentences, max_positions,
            ignore_invalid_inputs=True, allow_different_src_lens=True))

        if not sort_by_source_size:
            np.random.shuffle(batches)

        if sample:
            offset = (epoch - 1) * sample
            while offset > len(batches):
                np.random.shuffle(batches)
                offset -= len(batches)

            result = batches[offset:(offset + sample)]
            while len(result) < sample:
                np.random.shuffle(batches)
                result += batches[:(sample - len(result))]

            assert len(result) == sample, \
                "batch length is not correct {}".format(len(result))

            batches = result

        return batches

    def collater(self, samples):
        """Used by DataLoader. Merges a list of samples to form a mini-batch."""
        return self.collate(samples, self.pad_id, self.eos_id, self.trg is not None)

    @staticmethod
    def collate(samples, pad_id, eos_id, has_target=True):
        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False):
            return LanguagePairDataset.collate_tokens(
                [s[key] for s in samples],
                pad_id, eos_id, left_pad, move_eos_to_beginning,
            )

        id_ = th.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', left_pad=LanguagePairDataset.LEFT_PAD_SOURCE)
        # sort by descending source length
        src_lengths = th.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id_ = id_.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        trg_tokens = None
        trg_tokens_train = None
        trg_lengths = None
        ntokens = None
        if has_target:
            trg_tokens = merge('target', left_pad=LanguagePairDataset.LEFT_PAD_TARGET)
            trg_tokens = trg_tokens.index_select(0, sort_order)
            trg_tokens_train = merge(
                'target',
                left_pad=LanguagePairDataset.LEFT_PAD_TARGET,
                move_eos_to_beginning=True,     # Target for training, left shift.
            )
            trg_tokens_train = trg_tokens_train.index_select(0, sort_order)
            trg_lengths = th.LongTensor([s['target'].numel() for s in samples])
            ntokens = sum(len(s['target']) for s in samples)

        return {
            'id': id_,
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'trg_tokens': trg_tokens_train,
                'trg_lengths': trg_lengths,
            },
            'target': trg_tokens,
        }

    @staticmethod
    def collate_tokens(values, pad_id, eos_id, left_pad, move_eos_to_beginning=False):
        size = max(v.size(0) for v in values)
        res = values[0].new(len(values), size).fill_(pad_id)

        def copy_tensor(src, trg):
            assert trg.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_id
                trg[0] = eos_id
                trg[1:] = src[:-1]
            else:
                trg.copy_(src)

        for i, v in enumerate(values):
            if left_pad:
                copy_tensor(v, res[i][size - len(v):])
            else:
                copy_tensor(v, res[i][:len(v)])
        return res


def _valid_size(src_size, trg_size, max_positions):
    if isinstance(max_positions, numbers.Number):
        max_src_positions, max_trg_positions = max_positions, max_positions
    else:
        max_src_positions, max_trg_positions = max_positions
    if src_size < 1 or src_size > max_src_positions:
        return False
    if trg_size is not None and (trg_size < 1 or trg_size > max_trg_positions):
        return False
    return True


def _make_batches(src, trg, indices, max_tokens, max_sentences, max_positions,
                  ignore_invalid_inputs=False, allow_different_src_lens=False):
    batch = []

    def yield_batch(next_idx, num_tokens):
        if len(batch) == 0:
            return False
        if len(batch) == max_sentences:
            return True
        if num_tokens > max_tokens:
            return True
        if not allow_different_src_lens and \
                (src.sizes[batch[0]] != src.sizes[next_idx]):
            return True
        return False

    sample_len = 0
    ignored = []
    for idx in map(int, indices):
        src_size = src.sizes[idx]
        trg_size = trg.sizes[idx] if trg else src_size
        if not _valid_size(src_size, trg_size, max_positions):
            if ignore_invalid_inputs:
                ignored.append(idx)
                continue
            raise Exception((
                "Sample #{} has size (src={}, trg={}) but max size is {}."
                " Skip this example with --skip-invalid-size-inputs-valid-test"
            ).format(idx, src_size, trg_size, max_positions))

        sample_len = max(sample_len, src_size, trg_size)
        num_tokens = (len(batch) + 1) * sample_len
        if yield_batch(idx, num_tokens):
            yield batch
            batch = []
            sample_len = max(src_size, trg_size)

        batch.append(idx)

    if len(batch) > 0:
        yield batch

    if len(ignored) > 0:
        print("Warning! {} samples are either too short or too long "
              "and will be ignored, first few sample ids={}".format(len(ignored), ignored[:10]))


def mask_batches(batch_sampler, shard_id, num_shards):
    if num_shards == 1:
        return batch_sampler
    res = [
        batch
        for i, batch in enumerate(batch_sampler)
        if i % num_shards == shard_id
    ]
    expected_length = int(math.ceil(len(batch_sampler) / num_shards))
    return res + [[]] * (expected_length - len(res))
