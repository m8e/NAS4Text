#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import math
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

        self.stop_early = not hparams.no_early_stop
        self.normalize_scores = not hparams.unnormalized

    def _get_input_iter(self):
        itr = self.datasets.eval_dataloader(
            self.hparams.gen_subset,
            max_tokens=self.hparams.max_tokens,
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
        return self.decoding(beam=None)

    def beam_search(self):
        return self.decoding(beam=self.hparams.beam)

    def decoding(self, beam=None):
        itr = self._get_input_iter()

        gen_timer = StopwatchMeter()

        src_dict, trg_dict = self.datasets.source_dict, self.datasets.target_dict

        gen_subset_len = len(self.datasets.get_dataset(self.hparams.gen_subset))

        translated_strings = [None for _ in range(gen_subset_len)]
        for i, sample in enumerate(itr):
            if beam is None or beam <= 1:
                batch_translated_tokens = self._greedy_decoding(sample, gen_timer)
            else:
                batch_translated_tokens = self._beam_search_slow(sample, beam, gen_timer)
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

    def _get_maxlen(self, srclen):
        if self.hparams.use_task_maxlen:
            a, b = self.task.get_maxlen_a_b()
        else:
            a, b = self.hparams.maxlen_a, self.hparams.maxlen_b
        maxlen = max(1, int(a * srclen + b))
        return maxlen

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

        maxlen = self._get_maxlen(srclen)

        for model in self.models:
            model.eval()

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
                avg_probs, _ = self._decode(encoder_outs, input_['src_lengths'], trg_tokens, trg_lengths)

                if self.hparams.greedy_sample_temperature == 0.0:
                    _, next_word = avg_probs.max(dim=1)
                else:
                    assert self.hparams.greedy_sample_temperature > 0.0
                    next_word = th.multinomial(th.exp(avg_probs) / self.hparams.greedy_sample_temperature, 1)[:, 0]

                trg_tokens.data = th.cat([trg_tokens.data, th.unsqueeze(next_word, dim=1)], dim=1)
                trg_lengths += 1

        if timer is not None:
            timer.stop(batch_size)

        # Remove start tokens.
        return trg_tokens[:, 1:].data

    def _beam_search_slow(self, sample, beam, timer=None):
        sample = common.make_variable(sample, volatile=True, cuda=self.is_cuda)
        batch_size = sample['id'].numel()
        for model in self.models:
            model.eval()
        if timer is not None:
            timer.start()
        with common.maybe_no_grad():
            finalized = self._beam_search_slow_internal(sample, beam)
        if timer is not None:
            timer.stop(batch_size)

        # Postprocess beam search translations.
        # [NOTE]: Only return the top hypnosis with highest score.
        result = [hypos[0]['tokens'] for hypos in finalized]
        return result

    def _beam_search_slow_internal(self, sample, beam):
        batch_size = sample['id'].numel()
        input_ = sample['net_input']
        src_tokens = input_['src_tokens']
        srclen = src_tokens.size(1)
        start_symbol = self.task.EOS_ID
        vocab_size = self.task.get_vocab_size(is_src_lang=False)

        maxlen = self._get_maxlen(srclen)
        minlen = 1

        # Get prefix tokens.
        if self.hparams.prefix_size > 0:
            prefix_tokens = sample['target'].data[:, :self.hparams.prefix_size]
        else:
            prefix_tokens = None

        # compute the encoder output for each beam
        beam_src_tokens = input_['src_tokens'].repeat(1, beam).view(-1, srclen)
        beam_src_lengths = input_['src_lengths'].repeat(beam)
        encoder_outs = [
            model.encode(beam_src_tokens, beam_src_lengths)
            for model in self.models
        ]

        # buffers
        scores = src_tokens.data.new(batch_size * beam, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(batch_size * beam, maxlen + 2).fill_(self.task.PAD_ID)
        tokens_buf = tokens.clone()
        tokens[:, 0] = start_symbol
        tokens_var = common.make_variable(tokens, volatile=True, cuda=self.is_cuda)
        trg_lengths = common.make_variable(
            th.zeros(batch_size * beam).type_as(input_['src_lengths'].data),
            volatile=True, cuda=self.is_cuda)
        attn = scores.new(batch_size * beam, src_tokens.size(1), maxlen + 2)
        attn_buf = attn.clone()

        # list of completed sentences
        finalized = [[] for _ in range(batch_size)]
        finished = [False for _ in range(batch_size)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for _ in range(batch_size)]
        num_remaining_sent = batch_size

        # number of candidate hypos per step
        cand_size = 2 * beam    # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (th.arange(0, batch_size) * beam).unsqueeze(1).type_as(tokens)
        cand_offsets = th.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam
            if len(finalized[sent]) == beam:
                if self.stop_early or step == maxlen or unfinalized_scores is None:
                    return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= maxlen
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.task.EOS_ID
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2]

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.hparams.lenpen

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                sent = idx // beam
                sents_seen.add(sent)

                def get_hypo():
                    _, alignment = attn_clone[i].max(dim=0)
                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': attn_clone[i],  # src_len x tgt_len
                        'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent]['score']:
                    # replace worst hypo for this sentence with new/better one
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()

                    # find new worst finalized hypo for this sentence
                    idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                    worst_finalized[sent] = {
                        'score': s['score'],
                        'idx': idx,
                    }

            # return number of hypotheses finished this step
            num_finished = 0
            for sent in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    num_finished += 1
            return num_finished

        for step in range(maxlen + 1):  # one extra step for EOS marker
            # print('$', step, tokens_var.shape)
            # [NOTE]: Omitted: reorder decoder internal states based on the prev choice of beams

            trg_lengths[:] = step + 1
            # avg_probs: (batch_size * beam, vocab_size)
            probs, attn_scores = self._decode(
                encoder_outs, beam_src_lengths,
                common.make_variable(tokens[:, :step + 1], volatile=True, cuda=self.is_cuda), trg_lengths,
                compute_attn=True)

            if step == 0:
                # at the first step all hypotheses are equally likely, so use only the first beam
                # [NOTE]: Use ``unfold`` to get the first beam, see docstring.
                probs = probs.unfold(0, 1, beam).squeeze(2).contiguous()
                scores = scores.type_as(probs)
                scores_buf = scores_buf.type_as(probs)
            elif self.hparams.sampling:
                # make probs contain cumulative scores for each hypothesis
                probs.add_(scores[:, step - 1].view(-1, 1))

            probs[:, self.task.PAD_ID] = -math.inf  # never select pad
            probs[:, self.task.UNK_ID] -= self.hparams.unkpen  # apply unk penalty

            # Record attention scores
            attn[:, :, step + 1].copy_(attn_scores)

            cand_scores = buffer('cand_scores', type_of=scores)
            cand_indices = buffer('cand_indices')
            cand_beams = buffer('cand_beams')
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            if step < maxlen:
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    raise NotImplementedError('Prefix tokens not implemented')
                elif self.hparams.sampling:
                    raise NotImplementedError('Sampling not implemented')
                else:
                    # take the best 2 x beam_size predictions. We'll choose the first
                    # beam_size of these which don't predict eos to continue with.
                    th.topk(
                        probs.view(batch_size, -1),
                        k=min(cand_size, probs.view(batch_size, -1).size(1) - 1),   # -1 so we never select pad
                        out=(cand_scores, cand_indices),
                    )
                    th.div(cand_indices, vocab_size, out=cand_beams)
                    cand_indices.fmod_(vocab_size)
            else:
                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                th.sort(
                    probs[:, self.task.EOS_ID],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores)
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add_(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.task.EOS_ID)
            if step >= minlen:
                # only consider eos when it's among the top beam_size indices
                th.masked_select(
                    cand_bbsz_idx[:, :beam],
                    mask=eos_mask[:, :beam],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    th.masked_select(
                        cand_scores[:, :beam],
                        mask=eos_mask[:, :beam],
                        out=eos_scores,
                    )
                    num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            th.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            th.topk(
                active_mask, k=beam, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )
            active_bbsz_idx = buffer('active_bbsz_idx')
            th.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = th.gather(cand_scores, dim=1, index=active_hypos).view(-1)
            scores[:, step] = active_scores
            active_bbsz_idx = active_bbsz_idx.view(-1)

            # copy tokens and scores for active hypotheses
            th.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            th.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(batch_size, beam, -1)[:, :, step + 1],
            )
            if step > 0:
                th.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            th.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(batch_size, beam, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            th.index_select(
                attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                out=attn_buf[:, :, :step + 2],
            )

            # swap buffers
            old_tokens = tokens
            tokens = tokens_buf
            tokens_buf = old_tokens
            old_scores = scores
            scores = scores_buf
            scores_buf = old_scores
            old_attn = attn
            attn = attn_buf
            attn_buf = old_attn

            # [NOTE]: Omitted: reorder incremental state in decoder

        # sort by score descending
        for sent in range(batch_size):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)

        return finalized

    def _decode(self, encoder_outs, src_lengths, trg_tokens, trg_lengths, compute_attn=False):
        net_outputs = [
            model.decode(
                encoder_out, src_lengths,
                trg_tokens, trg_lengths)
            for encoder_out, model in zip(encoder_outs, self.models)
        ]
        return self._get_normalized_probs(net_outputs, compute_attn=compute_attn)

    def _get_normalized_probs(self, net_outputs, compute_attn=False, average_heads=True):
        avg_probs = None
        avg_attn = None
        for model, (output, attn) in zip(self.models, net_outputs):
            output = output[:, -1, :]
            probs = model.get_normalized_probs((output, avg_attn), log_probs=False).data
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)

            if not compute_attn:
                continue
            if attn is not None:
                if self.hparams.enc_dec_attn_type == 'fairseq':
                    attn = attn[:, -1, :].data
                elif self.hparams.enc_dec_attn_type == 'dot_product':
                    attn = attn[:, :, -1, :].data
                    if average_heads:
                        attn = attn.mean(dim=1)
                else:
                    raise ValueError('Unknown encoder-decoder attention type {}'.format(self.hparams.enc_dec_attn_type))
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs.div_(len(self.models))
        avg_probs.log_()

        if compute_attn and avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn


def generate_main(hparams, datasets=None):
    components = main_entry(hparams, datasets, train=False)

    # Check generator hparams
    assert hparams.path is not None, '--path required for generation!'
    assert not hparams.sampling or hparams.nbest == hparams.beam, '--sampling requires --nbest to be equal to --beam'

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
        logging.info('Use CUDA, running on device {}'.format(th.cuda.current_device()))

    if hparams.beam <= 0:
        generator.greedy_decoding()
    else:
        generator.beam_search()
