#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict
import contextlib
import logging
import os
import traceback

import numpy as np
import torch as th
from torch.autograd import Variable
from torch.serialization import default_restore_location

from . import UseFairseqParallel

__author__ = 'fyabc'


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return th.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def save_state(filename, hparams, model, criterion, optimizer, lr_scheduler,
               num_updates, optim_history=None, extra_state=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        'hparams': hparams,
        'model': model.state_dict(),
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'num_updates': num_updates,
            }
        ],
        'last_optimizer_state': optimizer.state_dict(),
        'extra_state': extra_state,
    }
    torch_persistent_save(state_dict, filename)


def load_model_state(filename, model, cuda_device=None):
    if not os.path.exists(filename):
        return None, [], None
    if cuda_device is None:
        state = th.load(filename)
    else:
        state = th.load(
            filename,
            map_location=lambda s, l: default_restore_location(s, 'cuda:{}'.format(cuda_device))
        )
    state = _upgrade_state_dict(state)
    state['model'] = model.upgrade_state_dict(state['model'])

    # load model parameters
    try:
        model.load_state_dict(state['model'])
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')

    return state['extra_state'], state['optimizer_history'], state['last_optimizer_state']


def _upgrade_state_dict(state):
    """Helper for upgrading old model checkpoints."""
    # add optimizer_history
    if 'optimizer_history' not in state:
        state['optimizer_history'] = [
            {
                'criterion_name': 'CrossEntropyCriterion',
                'best_loss': state['best_loss'],
            },
        ]
        state['last_optimizer_state'] = state['optimizer']
        del state['optimizer']
        del state['best_loss']
    # move extra_state into sub-dictionary
    if 'epoch' in state and 'extra_state' not in state:
        state['extra_state'] = {
            'epoch': state['epoch'],
            'batch_offset': state['batch_offset'],
            'val_loss': state['val_loss'],
        }
        del state['epoch']
        del state['batch_offset']
        del state['val_loss']
    # reduce optimizer history's memory usage (only keep the last state)
    if 'optimizer' in state['optimizer_history'][-1]:
        state['last_optimizer_state'] = state['optimizer_history'][-1]['optimizer']
        for optim_hist in state['optimizer_history']:
            del optim_hist['optimizer']
    # record the optimizer class name
    if 'optimizer_name' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['optimizer_name'] = 'FairseqNAG'
    # move best_loss into lr_scheduler_state
    if 'lr_scheduler_state' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['lr_scheduler_state'] = {
            'best': state['optimizer_history'][-1]['best_loss'],
        }
        del state['optimizer_history'][-1]['best_loss']
    # keep track of number of updates
    if 'num_updates' not in state['optimizer_history'][-1]:
        state['optimizer_history'][-1]['num_updates'] = 0
    return state


def maybe_no_grad(condition=True):
    if hasattr(th, 'no_grad') and condition:
        return th.no_grad()
    # no-op context manager
    return contextlib.ExitStack()


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def make_positions(tensor, padding_idx, left_pad):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """
    max_pos = padding_idx + 1 + tensor.size(1)

    # [NOTE]: Do not use range buffer if not use fairseq parallel,
    # because tensor and buffer must be on the same GPU device.
    if UseFairseqParallel:
        if not hasattr(make_positions, 'range_buf'):
            make_positions.range_buf = tensor.new()
        make_positions.range_buf = make_positions.range_buf.type_as(tensor)
        range_buf = make_positions.range_buf
    else:
        range_buf = tensor.new()

    if range_buf.numel() < max_pos:
        th.arange(padding_idx + 1, max_pos, out=range_buf)
    mask = tensor.ne(padding_idx)
    positions = range_buf[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    return tensor.clone().masked_scatter_(mask, positions[mask])


def mask_from_lengths(lengths, left_pad, cuda=False):
    if isinstance(lengths, Variable):
        lengths_ = lengths.data
    else:
        lengths_ = lengths
    batch_size = len(lengths_)
    max_length = max(lengths_)
    if cuda:
        tensor_type = th.cuda.ByteTensor
    else:
        tensor_type = th.ByteTensor
    result = tensor_type(batch_size, max_length).fill_(0)
    if isinstance(lengths, Variable):
        result = Variable(result, requires_grad=lengths.requires_grad)

    for i, length in enumerate(lengths_):
        if left_pad:
            result[i, max_length - length:] = 1
        else:
            result[i, :length] = 1

    return result


def volatile_variable(*args, **kwargs):
    if hasattr(th, 'no_grad'):
        # volatile has been deprecated, use the no_grad context manager instead
        return Variable(*args, **kwargs)
    else:
        return Variable(*args, **kwargs, volatile=True)


def make_variable(sample, volatile=False, cuda=False):
    """Wrap input tensors in Variable class."""

    if len(sample) == 0:
        return {}

    def _make_variable(maybe_tensor):
        if th.is_tensor(maybe_tensor):
            if cuda and th.cuda.is_available():
                maybe_tensor = maybe_tensor.cuda()
            if volatile:
                return volatile_variable(maybe_tensor)
            else:
                return Variable(maybe_tensor)
        elif isinstance(maybe_tensor, dict):
            return {
                key: _make_variable(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_make_variable(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _make_variable(sample)


INCREMENTAL_STATE_INSTANCE_ID = defaultdict(lambda: 0)


def _get_full_incremental_state_key(module_instance, key):
    module_name = module_instance.__class__.__name__

    # assign a unique ID to each module instance, so that incremental state is
    # not shared across module instances
    if not hasattr(module_instance, '_fairseq_instance_id'):
        INCREMENTAL_STATE_INSTANCE_ID[module_name] += 1
        module_instance._fairseq_instance_id = INCREMENTAL_STATE_INSTANCE_ID[module_name]

    return '{}.{}.{}'.format(module_name, module_instance._fairseq_instance_id, key)


def get_incremental_state(module, incremental_state, key):
    """Helper for getting incremental state for an nn.Module."""
    full_key = _get_full_incremental_state_key(module, key)
    if incremental_state is None or full_key not in incremental_state:
        return None
    return incremental_state[full_key]


def set_incremental_state(module, incremental_state, key, value):
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        full_key = _get_full_incremental_state_key(module, key)
        incremental_state[full_key] = value


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor
