#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Single process training functions."""

import collections
import itertools
import logging
import math
import os

import torch as th

from .utils.main_utils import main_entry
from .utils.data_processing import LanguageDatasets
from .models.child_net_base import ParalleledChildNet
from .criterions import build_criterion
from .child_trainer import ChildTrainer
from .utils.common import get_net_type
from .utils.paths import get_model_path
from .utils.meters import StopwatchMeter, AverageMeter
from .utils.progress_bar import build_progress_bar

__author__ = 'fyabc'


def single_process_main(hparams, datasets=None):
    """Main entry for training the child network.

    Can be called by user or the teacher model.

    Args:
        hparams: Hyper-parameters passed to the child trainer.
        datasets: Preload datasets, may be used to share dataset when called by teacher model.

    Returns:
        The child trainer instance.
    """

    # TODO: Like fairseq-py, add multiprocessing and distributed training.

    components = main_entry(hparams, datasets=datasets, train=True)
    net_code = components['net_code']
    datasets = components['datasets']

    # Build model and criterion
    model = get_net_type(net_code)(net_code, hparams)
    model = ParalleledChildNet(model, output_device=hparams.device_id)
    criterion = build_criterion(hparams, datasets.source_dict, datasets.target_dict)
    logging.info('Model structure:\n{}'.format(model))
    logging.info('Criterion: {}'.format(criterion.__class__.__name__))
    logging.info('All model parameters:')
    for name, param in model.named_parameters():
        logging.info('\t {}: {}, {}'.format(name, list(param.shape), param.numel()))
    logging.info('Number of parameters: {}'.format(model.num_parameters()))

    # Build trainer
    trainer = ChildTrainer(hparams, model, criterion)
    logging.info('Training on {} GPUs'.format(hparams.distributed_world_size))
    logging.info('Max tokens per GPU = {}, max sentences per GPU = {}'.format(
        hparams.max_tokens,
        hparams.max_sentences,
    ))

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

    # Send a dummy batch to warm the caching allocator
    # [DEBUG]: Dummy batch different because of random states.
    dummy_batch = datasets.get_dataset('train').get_dummy_batch(hparams.max_tokens, trainer.get_model().max_positions())
    trainer.dummy_train_step(dummy_batch)

    # Train until the learning rate gets too small
    max_epoch = hparams.max_epoch or math.inf
    max_update = hparams.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    while lr > hparams.min_lr and epoch <= max_epoch:
        # Train for one epoch
        train(hparams, trainer, datasets, epoch, batch_offset)

        # Evaluate on validate set
        if epoch % hparams.validate_interval == 0:
            for k, subset in enumerate(hparams.valid_subset.split(',')):
                val_loss = validate(hparams, trainer, datasets, subset, epoch)
                if k == 0:
                    # Only use first validation loss to update the learning schedule
                    lr = trainer.lr_step(epoch, val_loss)

                    # save checkpoint
                    if not hparams.no_save:
                        save_checkpoint(trainer, hparams, epoch, 0, val_loss)
        else:
            lr = trainer.lr_step(epoch)

        epoch += 1
        batch_offset = 0

        if trainer.get_num_updates() >= max_update:
            break
    train_meter.stop()
    logging.info('Training done in {:.1f} seconds'.format(train_meter.sum))


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

    # Reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    max_update = hparams.max_update or math.inf
    for i, sample in enumerate(progress, start=batch_offset):
        log_output = trainer.train_step(sample)

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
