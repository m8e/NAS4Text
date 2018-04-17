#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Dictionary: a simple wrapper of dict."""

from ..tasks import get_task

__author__ = 'fyabc'


class Dictionary:
    def __init__(self, dict_, task, is_src_lang=True):
        self._dict = dict_

        if isinstance(task, str):
            task = get_task(task)
        self.task = task
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

    def _check_dict(self):
        assert len(self._dict) == self.task.get_vocab_size(self.is_src_lang), 'Incorrect vocabulary size'

        assert self._dict.get(self.task.PAD, None) == self.task.PAD_ID, 'Incorrect PAD id'
        assert self._dict.get(self.task.EOS, None) == self.task.EOS_ID, 'Incorrect EOS id'
        assert self._dict.get(self.task.UNK, None) == self.task.UNK_ID, 'Incorrect UNK id'

    def __eq__(self, other):
        return self.task == other.dict and self._dict == other.dict

    def __len__(self):
        return len(self._dict)

    def get(self, symbol, add_if_not_exist=False):
        if add_if_not_exist:
            return self._dict.setdefault(symbol, len(self._dict))
        return self._dict.get(symbol, self.task.UNK_ID)
