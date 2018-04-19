#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import os

import torch as th

from .utils.main_utils import main_entry
from .utils.paths import get_model_path
from .utils.data_processing import ShardedIterator
from .utils.meters import StopwatchMeter
from .utils import common

__author__ = 'fyabc'


class ChildGenerator:
    def __init__(self, hparams, datasets, models):
        """

        Args:
            hparams: HParams object.
            datasets:
            models (list): List (ensemble) of models.
        """
        self.hparams = hparams
        self.datasets = datasets
        self.models = models
        self.is_cuda = False

    def _get_input_iter(self):
        itr = self.datasets.eval_dataloader(
            self.hparams.gen_subset,
            max_sentences=self.hparams.max_sentences,
            max_positions=min(model.max_encoder_positions() for model in self.models),
            skip_invalid_size_inputs_valid_test=self.hparams.skip_invalid_size_inputs_valid_test,
        )

        if self.hparams.num_shards > 1:
            if not (0 <= self.hparams.shard_id < self.hparams.num_shards):
                raise ValueError('--shard-id must be between 0 and num_shards')
            itr = ShardedIterator(itr, self.hparams.num_shards, self.hparams.shard_id)

        return itr

    def cuda(self):
        for model in self.models:
            model.cuda()
        self.is_cuda = True
        return self

    def greedy_decoding(self):
        itr = self._get_input_iter()

        gen_timer = StopwatchMeter()

        for sample in itr:
            print('#', sample)

    def beam_search(self):
        pass


def generate_main(hparams, datasets=None):
    # Check generator hparams
    assert hparams.path is not None, '--path required for generation!'
    assert not hparams.sampling or hparams.nbest == hparams.beam, '--sampling requires --nbest to be equal to --beam'

    components = main_entry(hparams, datasets, train=False)
    net_code = components['net_code']
    datasets = components['datasets']

    use_cuda = th.cuda.is_available() and not hparams.cpu

    # Load ensemble
    model_path = get_model_path(hparams)
    logging.info('Loading models from {}'.format(', '.join(hparams.path)))
    models, _ = common.load_ensemble_for_inference(
        [os.path.join(model_path, name) for name in hparams.path], net_code=net_code)

    # TODO: Optimize ensemble for generation
    # TODO: Load alignment dictionary for unknown word replacement

    # Build generator
    generator = ChildGenerator(hparams, datasets, models)
    if use_cuda:
        generator.cuda()

    generator.greedy_decoding()
