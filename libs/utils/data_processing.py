#! /usr/bin/python
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset

__author__ = 'fyabc'


class LanguagePairDataset(Dataset):
    # Padding constants
    LEFT_PAD_SOURCE = True
    LEFT_PAD_TARGET = False

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
