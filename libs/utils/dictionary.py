#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Dictionary: a simple wrapper of dict."""

import logging
import pickle

import torch as th

from ..tasks import get_task

__author__ = 'fyabc'


class Dictionary:
    def __init__(self, filename, task, is_src_lang=True, mode='pkl'):
        if isinstance(task, str):
            task = get_task(task)
        self.task = task

        self.mode = mode
        self.nspecial = None
        if mode == 'pkl':
            with open(filename, 'rb') as f:
                self._dict = pickle.load(f, encoding='utf-8')
        elif mode == 'text':
            with open(filename, 'r', encoding='utf-8') as f:
                self._dict = self._parse_text_dict(f)
        elif mode == 'empty':
            self._dict = self._default_symbols()
        else:
            raise ValueError('Unknown dictionary initialization mode {!r}'.format(mode))

        self._idict = {v: k for k, v in self._dict.items()}

        self.is_src_lang = is_src_lang

        self._check_dict()

    @property
    def dict(self):
        return self._dict

    @property
    def pad_id(self):
        return self.task.PAD_ID

    @property
    def eos_id(self):
        return self.task.EOS_ID

    @property
    def unk_id(self):
        return self.task.UNK_ID

    @property
    def language(self):
        if self.is_src_lang:
            return self.task.SourceLang
        return self.task.TargetLang

    def _default_symbols(self):
        result = {
            self.task.LUA: self.task.LUA_ID,
            self.task.PAD: self.task.PAD_ID,
            self.task.EOS: self.task.EOS_ID,
            self.task.UNK: self.task.UNK_ID,
        }
        self.nspecial = len(result)
        return result

    def _parse_text_dict(self, f):
        """Parse the text dict.

        [NOTE] Text dict format:
        File: dict.xx.de-en.de
        Content:
        word1 count1
        word2 count2
        ...
        word-n count-n

        Args:
            f: Text dict file.

        Returns:
            The dict.
        """
        # [NOTE]: Assume that the text dict does not contains special tokens.
        result = self._default_symbols()

        for line in f:
            words = line.strip().split()
            word = words[0]
            if word in result:
                logging.warning('Word {!r} already in dict, omitted'.format(word))
                continue
            result[word] = len(result)
        return result

    def _check_dict(self):
        if self.mode != 'empty':
            assert len(self._dict) == self.task.get_vocab_size(self.is_src_lang), \
                'Incorrect vocabulary size'

        assert self.nspecial == self.task.NumSpecialTokens, 'Incorrect number of special tokens'
        assert self._dict.get(self.task.PAD, None) == self.task.PAD_ID, 'Incorrect PAD id'
        assert self._dict.get(self.task.EOS, None) == self.task.EOS_ID, 'Incorrect EOS id'
        assert self._dict.get(self.task.UNK, None) == self.task.UNK_ID, 'Incorrect UNK id'

    def __eq__(self, other):
        return self.task == other.task and self._dict == other.dict

    def __len__(self):
        return len(self._dict)

    def get(self, symbol, add_if_not_exist=False):
        if add_if_not_exist:
            return self._dict.setdefault(symbol, len(self._dict))
        return self._dict.get(symbol, self.task.UNK_ID)

    def string(self, tensor, bpe_symbol=None, escape_unk=False, remove_pad=True):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if th.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)

        def yield_string(t):
            for elem in t:
                i = int(elem)   # Compatibility for PyTorch 0.3.1 and 0.4.0 (scalar tensor)
                if i == self.unk_id:
                    yield self.unk_string(escape_unk)
                elif i == self.pad_id:
                    if not remove_pad:
                        yield self._idict[i]
                elif i == self.eos_id:
                    # TODO: `continue` in fairseq-py, but should it be `return`?
                    return
                else:
                    yield self._idict[i]

        sent = ' '.join(yield_string(tensor))
        if bpe_symbol is not None:
            sent = sent.replace(bpe_symbol, '')
        return sent

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return '<{}>'.format(self.task.UNK)
        else:
            return self.task.UNK

    def dummy_sentence(self, length):
        t = th.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos_id
        return t

