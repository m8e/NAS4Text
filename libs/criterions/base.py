#! /usr/bin/python
# -*- coding: utf-8 -*-

from torch.nn.modules.loss import _Loss

__author__ = 'fyabc'


class BaseCriterion(_Loss):
    def __init__(self, hparams, src_dict, trg_dict):
        super().__init__()
        self.hparams = hparams
        self.padding_idx = trg_dict.pad_id

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def grad_denom(sample_sizes):
        """Compute the gradient denominator for a set of sample sizes."""
        return sum(sample_sizes)
