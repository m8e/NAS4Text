#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import os

import torch as th

from .utils.main_utils import main_entry
from .utils.paths import get_model_path, get_translate_output_path
from .utils.data_processing import ShardedIterator
from .utils.meters import StopwatchMeter
from .utils import common
from .tasks import get_task

__author__ = 'fyabc'


class ChildGenerator:
    def __init__(self, hparams, datasets, models, maxlen=None):
        """

        Args:
            hparams: HParams object.
            datasets:
            models (list): List (ensemble) of models.
        """
        self.hparams = hparams
        self.datasets = datasets
        self.task = get_task(hparams.task)
        self.models = models
        self.is_cuda = False

        max_decoder_len = min(m.max_decoder_positions() for m in self.models)
        max_decoder_len -= 1  # we define maxlen not including the EOS marker
        self.maxlen = max_decoder_len if maxlen is None else min(maxlen, max_decoder_len)

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

        src_dict, trg_dict = self.datasets.source_dict, self.datasets.target_dict

        translated_strings = []
        for i, sample in enumerate(itr):
            batch_translated_tokens = self._greedy_decoding(sample, gen_timer)
            print('Batch {}:'.format(i))
            for src_tokens, trg_tokens, translated_tokens in zip(
                    sample['net_input']['src_tokens'], sample['target'], batch_translated_tokens):
                print('SOURCE:', src_dict.string(src_tokens, bpe_symbol=self.hparams.remove_bpe))
                print('REF   :', trg_dict.string(trg_tokens, bpe_symbol=self.hparams.remove_bpe, escape_unk=True))
                trans_str = trg_dict.string(translated_tokens, bpe_symbol=self.hparams.remove_bpe, escape_unk=True)
                print('DECODE:', trans_str)
                translated_strings.append(trans_str)
            print()

        logging.info('Translated {} sentences in {:.1f}s ({:.2f} sentences/s)'.format(
            gen_timer.n, gen_timer.sum, 1. / gen_timer.avg))

        # Dump decoding outputs.
        if self.hparams.output_file is not None:
            output_path = get_translate_output_path(self.hparams)
            os.makedirs(output_path, exist_ok=True)
            full_path = os.path.join(output_path, self.hparams.output_file)
            with open(full_path, 'w') as f:
                for line in translated_strings:
                    print(line, file=f)
            logging.info('Decode output write to {}.'.format(full_path))

    def _greedy_decoding(self, sample, timer=None):
        """

        Args:
            sample (dict):
            timer (StopwatchMeter):
        """

        sample = common.make_variable(sample, volatile=True, cuda=self.is_cuda)
        batch_size = sample['id'].numel()
        input_ = sample['net_input']
        srclen = input_['src_tokens'].size(1)
        start_symbol = self.task.EOS_ID

        if self.hparams.use_task_maxlen:
            a, b = self.task.get_maxlen_a_b()
        else:
            a, b = self.hparams.maxlen_a, self.hparams.maxlen_b
        maxlen = int(a * srclen + b)

        if timer is not None:
            timer.start()

        with common.maybe_no_grad():
            encoder_outs = [
                model.encode(input_['src_tokens'], input_['src_lengths'])
                for model in self.models
            ]

            trg_tokens = common.make_variable(
                th.zeros(batch_size, 1).fill_(start_symbol).type_as(input_['src_tokens'].data),
                volatile=True, cuda=self.is_cuda)
            trg_lengths = common.make_variable(
                th.zeros(batch_size).fill_(1).type_as(input_['src_lengths'].data),
                volatile=True, cuda=self.is_cuda)
            for i in range(maxlen - 1):
                net_outputs = [
                    model.decode(
                        encoder_out, input_['src_lengths'],
                        trg_tokens, trg_lengths)
                    for encoder_out, model in zip(encoder_outs, self.models)
                ]

                avg_probs, _ = self._get_normalized_probs(net_outputs)
                _, next_word = avg_probs.max(dim=1)
                trg_tokens.data = th.cat([trg_tokens.data, th.unsqueeze(next_word, dim=1)], dim=1)
                trg_lengths += 1

        if timer is not None:
            timer.stop(batch_size)

        # Remove start tokens.
        return trg_tokens[:, 1:].data

    def beam_search(self):
        # TODO
        pass

    def _get_normalized_probs(self, net_outputs):
        avg_probs = None
        avg_attn = None
        for model, (output, attn) in zip(self.models, net_outputs):
            output = output[:, -1, :]
            probs = model.get_normalized_probs((output, avg_attn), log_probs=False).data
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)
            if attn is not None:
                attn = attn[:, :, -1, :].data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs.div_(len(self.models))
        avg_probs.log_()
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn


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
