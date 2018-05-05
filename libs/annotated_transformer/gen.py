#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import os

import torch

from .model import make_model
from .utils import fix_batch, subsequent_mask
from ..tasks import get_task
from ..utils import common
from ..utils.main_utils import main_entry
from ..utils.paths import get_model_path, get_translate_output_path
from ..utils.meters import StopwatchMeter
from ..utils.data_processing import ShardedIterator

__author__ = 'fyabc'


class SimpleGenerator:
    def __init__(self, hparams, datasets, model):
        self.hparams = hparams
        self.model = model
        self.datasets = datasets
        self.is_cuda = False
        self.task = get_task(self.hparams.task)

    def cuda(self):
        self.model.cuda()
        self.is_cuda = True
        return self

    def _get_input_iter(self):
        max_positions = self.hparams.max_src_positions - self.datasets.task.PAD_ID - 1
        itr = self.datasets.eval_dataloader(
            self.hparams.gen_subset,
            max_sentences=self.hparams.max_sentences,
            max_positions=max_positions,
            skip_invalid_size_inputs_valid_test=self.hparams.skip_invalid_size_inputs_valid_test,
        )

        if self.hparams.num_shards > 1:
            if not (0 <= self.hparams.shard_id < self.hparams.num_shards):
                raise ValueError('--shard-id must be between 0 and num_shards')
            itr = ShardedIterator(itr, self.hparams.num_shards, self.hparams.shard_id)

        return itr

    def greedy_decoding(self):
        itr = self._get_input_iter()

        gen_timer = StopwatchMeter()

        src_dict, trg_dict = self.datasets.source_dict, self.datasets.target_dict

        gen_subset_len = len(self.datasets.get_dataset(self.hparams.gen_subset))

        translated_strings = [None for _ in range(gen_subset_len)]
        for i, sample in enumerate(itr):
            batch_translated_tokens = self._greedy_decoding(sample, gen_timer)
            print('Batch {}:'.format(i))
            for id_, src_tokens, trg_tokens, translated_tokens in zip(
                    sample['id'], sample['net_input']['src_tokens'], sample['target'], batch_translated_tokens):
                print('SOURCE:', src_dict.string(src_tokens, bpe_symbol=self.task.BPESymbol))
                print('REF   :', trg_dict.string(trg_tokens, bpe_symbol=self.task.BPESymbol, escape_unk=True))
                trans_str = trg_dict.string(translated_tokens, bpe_symbol=self.task.BPESymbol, escape_unk=True)
                print('DECODE:', trans_str)
                translated_strings[id_] = trans_str
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
                    assert line is not None, 'There is a sentence not being translated'
                    print(line, file=f)
            logging.info('Decode output write to {}.'.format(full_path))

    def _greedy_decoding(self, sample, timer=None):
        fix_batch(sample, pad_id=self.datasets.task.PAD_ID)
        sample = common.make_variable(sample, volatile=True, cuda=self.is_cuda)
        batch_size = sample['id'].numel()
        input_ = sample['net_input']
        srclen = input_['src_tokens'].size(1)
        start_symbol = self.task.EOS_ID

        if self.hparams.use_task_maxlen:
            a, b = self.task.get_maxlen_a_b()
        else:
            a, b = self.hparams.maxlen_a, self.hparams.maxlen_b
        maxlen = max(1, int(a * srclen + b))

        self.model.eval()

        if timer is not None:
            timer.start()

        with common.maybe_no_grad():
            encoder_out = self.model.encode(input_['src_tokens'], input_['src_mask'])

            trg_tokens = common.make_variable(
                torch.zeros(batch_size, 1).fill_(start_symbol).type_as(input_['src_tokens'].data),
                volatile=True, cuda=self.is_cuda)
            trg_mask = common.make_variable(
                subsequent_mask(trg_tokens.size(1)),
                volatile=True, cuda=self.is_cuda)
            for i in range(maxlen - 1):
                net_output = self.model.decode(
                    encoder_out, input_['src_mask'], trg_tokens, trg_mask)

                prob = self.model.generator(net_output[:, -1])
                _, next_word = torch.max(prob, dim=1)

                trg_tokens.data = torch.cat([trg_tokens.data, torch.unsqueeze(next_word.data, dim=1)], dim=1)

        if timer is not None:
            timer.stop(batch_size)

        # Remove start tokens.
        return trg_tokens[:, 1:].data


def load_model(hparams, datasets):
    assert len(hparams.path) == 1, 'Only support single model generation for simple'
    save_dir = get_model_path(hparams)
    filename = os.path.join(save_dir, hparams.path[0])
    state = torch.load(
        filename,
        map_location=lambda s, l: common.default_restore_location(s, 'cpu')
    )

    model = make_model(
        hparams=hparams,
        src_vocab=datasets.task.get_vocab_size(is_src_lang=True),
        tgt_vocab=datasets.task.get_vocab_size(is_src_lang=False),
        N=2,
        d_model=256,
        d_ff=1024,
        h=8,
        dropout=0.1,
    )
    model.load_state_dict(state['model'])

    logging.info('Load checkpoint from {} (epoch {})'.format(hparams.path[0], state['extra_state']['epoch']))
    return model


def annotated_gen_main(hparams):
    hparams.net_code_file = 'annotated_transformer'

    components = main_entry(hparams, train=False, net_code=False)
    datasets = components['datasets']
    use_cuda = torch.cuda.is_available() and not hparams.cpu

    model = load_model(hparams, datasets)
    generator = SimpleGenerator(hparams, datasets, model)

    if use_cuda:
        generator.cuda()

    generator.greedy_decoding()
