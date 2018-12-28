#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Utilities for main entries."""

import collections
import itertools
import json
import logging
import math
import os
import pprint

import torch as th

from .paths import get_model_path
from ..hparams import get_hparams
from ..layers.net_code import get_net_code
from ..utils.data_processing import LanguageDatasets
from ..utils.progress_bar import build_progress_bar
from ..utils.meters import AverageMeter

__author__ = 'fyabc'


def _set_default_hparams(hparams):
    """Set default value of hparams.

    [NOTE]: Only set default value for hparams that default value in args is ``None``.
    """
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
            :key net_code (bool or str): Get net code or not. (True)
                If it is a string, use this string as net code filename prefix to store models and translated outputs.
            :key hparams_ppp (callable): HParams postprocessor.

    Returns:
        dict: Contains several components.
    """

    logging.basicConfig(
        format='[{levelname:<8}] {asctime}.{msecs:0>3.0f}: <{filename}:{lineno}> {message}',
        level=hparams.logging_level,
        style='{',
    )

    train_ = kwargs.pop('train', True)
    title = 'training' if train_ else 'generation'

    logging.info('Start single node {}'.format(title))
    logging.info('Task: {}'.format(hparams.task))
    logging.info('HParams set: {}'.format(hparams.hparams_set))

    # Get net code.
    # [NOTE]: Must before hparams postprocessing because of the hparams priority.
    net_code = kwargs.pop('net_code', True)
    if net_code is True:
        code = get_net_code(hparams, modify_hparams=True)
        logging.info('Net code:\n{}'.format(json.dumps(code.original_code, indent=4)))
    else:
        code = None
        if isinstance(net_code, str):
            hparams.net_code_file = net_code + '_' + hparams.net_code_file

    # Postprocess hparams.
    _set_default_hparams(hparams)
    if hasattr(hparams, 'update_freq'):
        hparams.update_freq = list(map(int, hparams.update_freq.split(',')))
    if train_:
        hparams.lr = list(map(float, hparams.lr.split(',')))
        if hparams.max_sentences_valid is None:
            hparams.max_sentences_valid = hparams.max_sentences
        hparams_ppp = kwargs.pop('hparams_ppp', None)
        if hparams_ppp is not None:
            hparams_ppp(hparams)

    logging.info('Child {} hparams:\n{}'.format(title, pprint.pformat(hparams.__dict__)))
    if train_:
        model_path = get_model_path(hparams)
        os.makedirs(model_path, exist_ok=True)
        hparams_path = os.path.join(get_model_path(hparams), 'hparams.json')
        with open(hparams_path, 'w', encoding='utf-8') as f:
            json.dump(hparams.__dict__, f, indent=4)
            logging.info('Dump hparams into {}'.format(hparams_path))
    logging.info('Search space information:')
    logging.info('LSTM search space: {}'.format(hparams.lstm_space))
    logging.info('Convolutional search space: {}'.format(hparams.conv_space))
    logging.info('Attention search space: {}'.format(hparams.attn_space))

    if train_:
        if not th.cuda.is_available():
            raise RuntimeError('Want to training on GPU but CUDA is not available')
        th.cuda.set_device(hparams.device_id)
        th.manual_seed(hparams.seed)

    # Load datasets
    if kwargs.pop('load_datasets', True):
        datasets = kwargs.pop('datasets', None)
        datasets = LanguageDatasets(hparams) if datasets is None else datasets
        logging.info('Dataset information:')
        _d_src = datasets.source_dict
        logging.info('Source dictionary [{}]: len = {}'.format(_d_src.language, len(_d_src)))
        _d_trg = datasets.target_dict
        logging.info('Source dictionary [{}]: len = {}'.format(_d_trg.language, len(_d_trg)))

        splits = ['train', 'dev'] if train_ else [hparams.gen_subset]
        # splits = ['test', 'dev'] if train_ else [hparams.gen_subset]    # [DEBUG]
        datasets.load_splits(splits)
        for split in splits:
            logging.info('Split {}: len = {}'.format(split, len(datasets.splits[split])))
    else:
        datasets = None

    return {
        'net_code': code,
        'datasets': datasets,
    }


def logging_model_criterion(model, criterion, logging_params=True):
    if logging_params:
        logging.info('Model structure:\n{}'.format(model))
    else:
        logging.info('Model: {}'.format(model.__class__.__name__))
    logging.info('Criterion: {}'.format(criterion.__class__.__name__))
    if logging_params:
        logging.info('All model parameters:')
        for name, param in model.named_parameters():
            logging.info('\t {}: {}, {}'.format(name, list(param.shape), param.numel()))
    logging.info('Number of parameters: {}'.format(model.num_parameters()))


def logging_training_stats(hparams):
    logging.info('Training on {} GPUs'.format(hparams.distributed_world_size))
    logging.info('Max tokens per GPU = {}, max sentences per GPU = {}'.format(
        hparams.max_tokens,
        hparams.max_sentences,
    ))


def prepare_checkpoint(hparams, trainer):
    # Load the latest checkpoint if one is available
    model_path = get_model_path(hparams)
    os.makedirs(model_path, exist_ok=True)
    checkpoint_path = os.path.join(model_path, hparams.restore_file)
    extra_state = trainer.load_checkpoint(checkpoint_path)
    if extra_state is not None:
        epoch = extra_state['epoch']
        batch_offset = extra_state['batch_offset']
        logging.info('Loaded checkpoint {} (epoch {})'.format(checkpoint_path, epoch))
        if batch_offset == 0:
            trainer.lr_step(epoch)
            epoch += 1
    else:
        logging.info('No loaded checkpoint, start from scratch')
        epoch, batch_offset = 1, 0

    return epoch, batch_offset


def train(hparams, trainer, datasets, epoch, batch_offset):
    """Train the model for one epoch.

    Args:
        hparams:
        trainer (ChildTrainer):
        datasets (LanguageDatasets):
        epoch (int):
        batch_offset (int):

    Returns:

    """

    # Set seed based on args.seed and the epoch number so that we get
    # reproducible results when resuming from checkpoints
    seed = hparams.seed
    th.manual_seed(seed)

    # The max number of positions can be different for train and valid
    # e.g., RNNs may support more positions at test time than seen in training
    max_positions_train = (
        min(hparams.max_src_positions, trainer.get_model().max_encoder_positions()),
        min(hparams.max_trg_positions, trainer.get_model().max_decoder_positions()),
    )

    # Initialize dataloader, starting at batch_offset
    itr = datasets.train_dataloader(
        hparams.train_subset,
        max_tokens=hparams.max_tokens,
        max_sentences=hparams.max_sentences,
        max_positions=max_positions_train,
        seed=seed,
        epoch=epoch,
        sample_without_replacement=hparams.sample_without_replacement,
        sort_by_source_size=(epoch <= hparams.curriculum),
        shard_id=hparams.distributed_rank,
        num_shards=hparams.distributed_world_size,
    )

    next(itertools.islice(itr, batch_offset, batch_offset), None)
    progress = build_progress_bar(hparams, itr, epoch, no_progress_bar='simple')

    # update parameters every N batches
    if epoch <= len(hparams.update_freq):
        update_freq = hparams.update_freq[epoch - 1]
    else:
        update_freq = hparams.update_freq[-1]

    # Reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    max_update = hparams.max_update or math.inf
    num_batches = len(itr)
    for i, sample in enumerate(progress, start=batch_offset):
        if i < num_batches - 1 and (i + 1) % update_freq > 0:
            trainer.train_step(sample, update_params=False)
            continue
        else:
            log_output = trainer.train_step(sample, update_params=True)

        # Log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

        if i == batch_offset:
            # Ignore the first mini-batch in words-per-second calculation
            trainer.get_meter('wps').reset()

        # save mid-epoch checkpoints
        num_updates = trainer.get_num_updates()
        if hparams.save_interval > 0 and num_updates > 0 and num_updates % hparams.save_interval == 0:
            save_checkpoint(trainer, hparams, epoch, i + 1)

        if num_updates >= max_update:
            break

    # Log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = '{:.3f}'.format(trainer.get_meter('train_loss').avg)
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss').avg
        stats['nll_loss'] = '{:.3f}'.format(nll_loss)
    else:
        nll_loss = trainer.get_meter('train_loss').avg
    stats['ppl'] = get_perplexity(nll_loss)
    stats['wps'] = round(trainer.get_meter('wps').avg)
    stats['ups'] = '{:.1f}'.format(trainer.get_meter('ups').avg)
    stats['wpb'] = round(trainer.get_meter('wpb').avg)
    stats['bsz'] = round(trainer.get_meter('bsz').avg)
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = '{:.3f}'.format(trainer.get_meter('gnorm').avg)
    stats['clip'] = '{:.0%}'.format(trainer.get_meter('clip').avg)
    stats['oom'] = trainer.get_meter('oom').avg
    return stats


def validate(hparams, trainer, datasets, subset, epoch):
    """Evaluate the model on the validation set and return the average loss.

    Args:
        hparams:
        trainer (ChildTrainer):
        datasets (LanguageDatasets):
        subset (str):
        epoch (int):

    Returns:

    """

    # Initialize dataloader
    max_positions_valid = (
        trainer.get_model().max_encoder_positions(),
        trainer.get_model().max_decoder_positions(),
    )
    itr = datasets.eval_dataloader(
        subset,
        max_tokens=hparams.max_tokens,
        max_sentences=hparams.max_sentences_valid,
        max_positions=max_positions_valid,
        skip_invalid_size_inputs_valid_test=hparams.skip_invalid_size_inputs_valid_test,
        descending=True,  # largest batch first to warm the caching allocator
        shard_id=hparams.distributed_rank,
        num_shards=hparams.distributed_world_size,
    )

    progress = build_progress_bar(
        hparams, itr, epoch,
        prefix='valid on \'{}\' subset'.format(subset),
        no_progress_bar='simple'
    )

    # Reset validation loss meters
    for k in ['valid_loss', 'valid_nll_loss']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    for sample in progress:
        log_output = trainer.valid_step(sample)

        # log mid-validation stats
        stats = get_valid_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss']:
                continue
            extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats)

    # log validation stats
    stats = get_valid_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats)

    return stats['valid_loss']


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['valid_loss'] = trainer.get_meter('valid_loss').avg
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss').avg
        stats['valid_nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('valid_loss').avg
    stats['valid_ppl'] = get_perplexity(nll_loss)
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(trainer, hparams, epoch, batch_offset, val_loss=None):
    extra_state = {
        'epoch': epoch,
        'batch_offset': batch_offset,
        'val_loss': val_loss,
    }

    save_dir = get_model_path(hparams)

    if batch_offset == 0:
        if not hparams.no_epoch_checkpoints:
            epoch_filename = os.path.join(save_dir, 'checkpoint{}.pt'.format(epoch))
            trainer.save_checkpoint(epoch_filename, extra_state)
            logging.info('Save checkpoint to {} (epoch {})'.format(epoch_filename, epoch))

        assert val_loss is not None
        if not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best:
            save_checkpoint.best = val_loss
            best_filename = os.path.join(save_dir, 'checkpoint_best.pt')
            trainer.save_checkpoint(best_filename, extra_state)
            logging.info('Save checkpoint to {} (epoch {})'.format(best_filename, epoch))
    elif not hparams.no_epoch_checkpoints:
        epoch_filename = os.path.join(
            save_dir, 'checkpoint{}_{}.pt'.format(epoch, batch_offset))
        trainer.save_checkpoint(epoch_filename, extra_state)
        logging.info('Save checkpoint to {} (epoch {})'.format(epoch_filename, epoch))

    last_filename = os.path.join(save_dir, 'checkpoint_last.pt')
    trainer.save_checkpoint(last_filename, extra_state)
    logging.info('Save checkpoint to {} (epoch {})'.format(last_filename, epoch))
