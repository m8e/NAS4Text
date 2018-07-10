#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Utilities for main entries."""

import logging
import pprint

import torch as th

from ..hparams import get_hparams
from ..layers.net_code import get_net_code
from ..utils.data_processing import LanguageDatasets

__author__ = 'fyabc'


def _set_default_hparams(hparams):
    """Set default value of hparams."""
    base_hparams = get_hparams(hparams.hparams_set)

    for name, value in base_hparams.__dict__.items():
        if getattr(hparams, name, None) is None:
            setattr(hparams, name, value)

    return hparams


def main_entry(hparams, **kwargs):
    """General code of main entries.

    Args:
        hparams:
        kwargs: Other keywords.
            :key load_dataset (bool): Load dataset or not. (True)
            :key datasets (LanguageDatasets): Preload datasets or None. (None)
            :key train (bool): In training or generation. (True)
            :key net_code (bool): Get net code or not. (True)

    Returns:
        dict: Contains several components.
    """

    logging.basicConfig(
        format='[{levelname:<8}] {asctime}.{msecs:0>3.0f}: <{filename}:{lineno}> {message}',
        level=hparams.logging_level,
        style='{',
    )

    train = kwargs.pop('train', True)
    title = 'training' if train else 'generation'

    logging.info('Start single node {}'.format(title))
    logging.info('Task: {}'.format(hparams.task))
    logging.info('HParams set: {}'.format(hparams.hparams_set))

    # Get net code.
    # [NOTE]: Must before hparams postprocessing because of the hparams priority.
    if kwargs.pop('net_code', True):
        code = get_net_code(hparams, modify_hparams=True)
    else:
        code = None

    # Postprocess hparams.
    _set_default_hparams(hparams)
    if train:
        hparams.lr = list(map(float, hparams.lr.split(',')))
        if hparams.max_sentences_valid is None:
            hparams.max_sentences_valid = hparams.max_sentences

    logging.info('Child {} hparams:\n{}'.format(title, pprint.pformat(hparams.__dict__)))
    logging.info('Search space information:')
    logging.info('LSTM search space: {}'.format(hparams.lstm_space))
    logging.info('Convolutional search space: {}'.format(hparams.conv_space))
    logging.info('Attention search space: {}'.format(hparams.attn_space))

    if train:
        if not th.cuda.is_available():
            raise RuntimeError('Want to training on GPU but CUDA is not available')
        th.cuda.set_device(hparams.device_id)
        th.manual_seed(hparams.seed)

    # Load datasets
    if kwargs.pop('load_datasets', True):
        datasets = kwargs.pop('datasets', None)
        datasets = LanguageDatasets(hparams.task) if datasets is None else datasets
        logging.info('Dataset information:')
        _d_src = datasets.source_dict
        logging.info('Source dictionary [{}]: len = {}'.format(_d_src.language, len(_d_src)))
        _d_trg = datasets.target_dict
        logging.info('Source dictionary [{}]: len = {}'.format(_d_trg.language, len(_d_trg)))

        splits = ['train', 'dev'] if train else [hparams.gen_subset]
        datasets.load_splits(splits)
        for split in splits:
            logging.info('Split {}: len = {}'.format(split, len(datasets.splits[split])))
    else:
        datasets = None

    return {
        'net_code': code,
        'datasets': datasets,
    }
