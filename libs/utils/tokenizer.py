#! /usr/bin/python
# -*- coding: utf-8 -*-

import re

import torch

__author__ = 'fyabc'


SPACE_NORMALIZER = re.compile("\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Tokenizer:
    @staticmethod
    def tokenize(line, dictionary, line_tokenizer=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.LongTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            idx = dictionary.get(word, add_if_not_exist=add_if_not_exist)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dictionary.eos_id
        return ids

