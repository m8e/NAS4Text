#! /usr/bin/python
# -*- coding: utf-8 -*-

from collections import OrderedDict, defaultdict
import logging
import math

import torch as th
import torch.nn as nn

from .optimizers import build_optimizer
from .optimizers.lr_schedulers import build_lr_scheduler
from .utils import common, distributed_utils, UseFairseqParallel
from .utils.meters import AverageMeter, TimeMeter

__author__ = 'fyabc'


class ChildTrainer:
    """Main class for multi-GPU training.

    Each GPU has a full copy of the model and is assigned to its own Python
    process. Gradients are accumulated with torch.distributed.all_reduce and all
    model replicas are updated synchronously after each batch.
    """
    def __init__(self, hparams, model, criterion):
        """

        Args:
            hparams:
            model:
            criterion:
        """
        if not th.cuda.is_available():
            raise NotImplementedError('Training on CPU is not supported')

        self.hparams = hparams

        # Copy model and criterion to current device
        if model is not None:
            self.model = model.cuda()
        else:
            self.model = None
        self.criterion = criterion.cuda()

        # Initialize optimizer and LR scheduler
        if model is not None:
            self.optimizer = build_optimizer(self.hparams, self.model.parameters())
            self.lr_scheduler = build_lr_scheduler(self.hparams, self.optimizer)
            logging.info('Optimizer: {}'.format(self.optimizer.__class__.__name__))
            logging.info('LR Scheduler: {}'.format(self.lr_scheduler.__class__.__name__))

        # Initialize meters
        self.meters = OrderedDict()
        self._init_meters()

        # [NOTE]: Different from fairseq: Does not use buffered stats
        # self._buffered_stats = defaultdict(lambda: [])

        self._max_bsz_seen = 0
        self._num_updates = 0

    def _init_meters(self):
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()  # words per second
        self.meters['ups'] = TimeMeter()  # updates per second
        self.meters['wpb'] = AverageMeter()  # words per batch
        self.meters['bsz'] = AverageMeter()  # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()  # % of updates clipped
        self.meters['oom'] = AverageMeter()  # out of memory

        self._flat_grads = None

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.hparams):
            extra_state['train_meters'] = self.meters
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            common.save_state(filename, self.hparams, model, self.criterion, self.optimizer,
                              self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
                              model.net_code)

    def load_checkpoint(self, filename):
        """Load all training state from a checkpoint file."""

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        extra_state, self._optim_history, last_optim_state = common.load_model_state(
            filename, model, cuda_device=th.cuda.current_device())

        if last_optim_state is not None:
            # rebuild optimizer after loading model, since params may have changed
            self.optimizer = build_optimizer(self.hparams, model.parameters())
            self.lr_scheduler = build_lr_scheduler(self.hparams, self.optimizer)

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            if last_optim['criterion_name'] == self.criterion.__class__.__name__:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
                if last_optim['optimizer_name'] == self.optimizer.__class__.__name__:
                    self.optimizer.load_state_dict(last_optim_state)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters = extra_state['train_meters']
            del extra_state['train_meters']

        return extra_state

    def train_step(self, sample, update_params=True):
        """Do forward, backward and parameter update."""

        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.hparams.seed + self.get_num_updates()
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)

        sample = self._prepare_sample(sample, volatile=False)

        # forward pass
        loss, sample_sizes, logging_outputs, ooms_fwd = self._forward(sample)

        # aggregate stats and logging outputs
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        grad_denom = self.criterion.__class__.grad_denom(sample_sizes)
        agg_logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)

        # backward pass, all-reduce gradients and take an optimization step
        grad_norm, ooms_bwd = self._backward_and_opt(loss, grad_denom, update_params=update_params)

        if grad_norm is None:
            return None

        # update meters
        self.meters['wps'].update(ntokens)
        self.meters['ups'].update(1.)
        self.meters['wpb'].update(ntokens)
        self.meters['bsz'].update(nsentences)
        self.meters['gnorm'].update(grad_norm)
        self.meters['clip'].update(1. if grad_norm > self.hparams.clip_norm else 0.)
        self.meters['oom'].update(ooms_fwd + ooms_bwd)

        # update loss meters for training
        if 'loss' in agg_logging_output:
            self.meters['train_loss'].update(agg_logging_output['loss'], grad_denom)
        # criterions can optionally log the NLL loss too
        if 'nll_loss' in agg_logging_output:
            self.meters['train_nll_loss'].update(agg_logging_output['nll_loss'], ntokens)

        return agg_logging_output

    def _forward(self, sample, eval_=False):
        if eval_:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        loss = None
        sample_size = 0
        logging_output = {
            'ntokens': sample['ntokens'] if sample is not None else 0,
            'nsentences': sample['target'].size(0) if sample is not None else 0,
        }
        oom = 0
        if sample is not None:
            try:
                with common.maybe_no_grad(eval_):
                    # calculate loss and sample size
                    loss, sample_size, logging_output_ = self.criterion(self.model, sample)
                    logging_output.update(logging_output_)
            except RuntimeError as e:
                if not eval_ and 'out of memory' in str(e):
                    logging.warning('Ran out of memory, skipping batch')
                    oom = 1
                    loss = None
                    if hasattr(th.cuda, 'empty_cache'):
                        th.cuda.empty_cache()
                else:
                    raise e

        # synchronize logging outputs for multi-GPU training
        if UseFairseqParallel and self.hparams.distributed_world_size > 1:
            sample_sizes, logging_outputs, ooms = zip(*list(
                distributed_utils.all_gather_list((sample_size, logging_output, oom))))
            ooms = sum(ooms)
        else:
            sample_sizes = [sample_size]
            logging_outputs = [logging_output]
            ooms = oom

        return loss, sample_sizes, logging_outputs, ooms

    def _backward_and_opt(self, loss, grad_denom, update_params=True):
        oom = 0
        if loss is not None:
            try:
                # backward pass
                loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    logging.warning('Ran out of memory, skipping batch')
                    oom = 1
                    if hasattr(th.cuda, 'empty_cache'):
                        th.cuda.empty_cache()
                    self.optimizer.zero_grad()
                else:
                    raise e

        if not update_params:
            return None, None

        grad_norm = self._all_reduce_and_scale(grad_denom)
        self._opt()

        return grad_norm, oom

    def _all_reduce_and_scale(self, grad_denom):
        # flatten grads into a single buffer and all-reduce
        flat_grads = self._flat_grads = self._get_flat_grads(self._flat_grads)
        if UseFairseqParallel and self.hparams.distributed_world_size > 1:
            th.distributed.all_reduce(flat_grads)

        # rescale and clip gradients
        # FIXME: Sum of flat grads same before here, different after here (why???)
        flat_grads.div_(grad_denom)
        grad_norm = common.clip_grad_norm_(flat_grads, self.hparams.clip_norm)

        # copy grads back into model parameters
        self._set_flat_grads(flat_grads)

        return grad_norm

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                   'Use the param in the forward pass or set requires_grad=False')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset+numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]

    def _set_flat_grads(self, new_grads):
        grads = self._get_grads()
        offset = 0
        for g in grads:
            numel = g.numel()
            g.copy_(new_grads[offset:offset+numel].view_as(g))
            offset += numel

    def _opt(self):
        # take an optimization step
        self.optimizer.step()
        self.zero_grad()
        self._num_updates += 1

        # update learning rate
        self.lr_scheduler.step_update(self._num_updates)

    def valid_step(self, sample):
        """Do forward pass in evaluation mode."""
        sample = self._prepare_sample(sample, volatile=True)

        # forward pass
        loss, sample_sizes, logging_outputs, ooms_fwd = self._forward(sample, eval_=True)
        assert not ooms_fwd, 'Ran out of memory during validation'

        # aggregate stats and logging outputs
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        grad_denom = self.criterion.__class__.grad_denom(sample_sizes)
        agg_logging_output = self.criterion.__class__.aggregate_logging_outputs(logging_outputs)

        # update loss meters for validation
        if 'loss' in agg_logging_output:
            self.meters['valid_loss'].update(agg_logging_output['loss'], grad_denom)
        # criterions can optionally log the NLL loss too
        if 'nll_loss' in agg_logging_output:
            self.meters['valid_nll_loss'].update(agg_logging_output['nll_loss'], ntokens)

        return agg_logging_output

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, update_params=False)
        self.zero_grad()

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the model replica."""
        return self.model

    def get_meter(self, name):
        """Get a specific meter by name."""
        return self.meters.get(name, None)

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample, volatile):
        if sample is None or len(sample) == 0:
            return None
        if hasattr(th.cuda, 'empty_cache'):
            # Clear the caching allocator if this is the largest sample we've seen
            if sample['target'].size(0) > self._max_bsz_seen:
                self._max_bsz_seen = sample['target'].size(0)
                th.cuda.empty_cache()

        return common.make_variable(sample, volatile=volatile, cuda=True)
