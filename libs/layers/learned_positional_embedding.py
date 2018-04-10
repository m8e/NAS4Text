#! /usr/bin/python
# -*- coding: utf-8 -*-

from torch.autograd import Variable
import torch.nn as nn

from ..utils.common import make_positions

__author__ = 'fyabc'


class LearnedPositionalEmbedding(nn.Embedding):
    """This module learns positional embeddings up to a fixed maximum size.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, left_pad):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.left_pad = left_pad

    def forward(self, input_, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen]."""
        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            positions = input_.data.new(1, 1).fill_(self.padding_idx + input_.size(1))
        else:
            positions = make_positions(input_.data, self.padding_idx, self.left_pad)
        return super().forward(Variable(positions))

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.num_embeddings - self.padding_idx - 1


__all__ = ['LearnedPositionalEmbedding']
