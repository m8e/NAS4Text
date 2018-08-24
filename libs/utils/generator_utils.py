#! /usr/bin/python
# -*- coding: utf-8 -*-

import collections
import logging
import math

import torch

from . import tokenizer

__author__ = 'fyabc'


def _trim(seq, token):
    start = 0
    length = len(seq)
    end = length - 1
    while seq[start] == token and start < length:
        start += 1
    while seq[end] == token and end >= 0 and end > start:
        end -= 1
    return seq[start:end + 1]


def _normalize(maybe_tensor, eos=None, pad=None, unk=None):
    # Unwrap tensor.
    if isinstance(maybe_tensor, torch.Tensor):
        result = maybe_tensor.tolist()
    else:
        result = maybe_tensor

    # Remove EOS.
    if eos is not None:
        result = _trim(result, eos)
    if pad is not None:
        result = _trim(result, pad)
    if unk is not None:
        result = [x if x != unk else -x for x in result]
    return result


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
            methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False, **kwargs):
    """Computes BLEU score of translated segments against one or more references.

    Copied from TensorFlow <https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py>.

    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
        kwargs: For compatibility.
    Returns:
        # 3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
        # precisions and brevity penalty.
        A float of BLEU score.
    """

    dict_ = kwargs.pop('dict', None)

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        normalized_translation = _normalize(translation, eos=dict_.eos_id, pad=dict_.pad_id, unk=dict_.unk_id)
        translation_length += len(normalized_translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            normalized_reference = _normalize(reference, eos=dict_.eos_id, pad=dict_.pad_id, unk=dict_.unk_id)
            reference_length += len(normalized_reference)
            merged_ref_ngram_counts |= _get_ngrams(normalized_reference, max_order)
        translation_ngram_counts = _get_ngrams(normalized_translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(normalized_translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    # return bleu, precisions, bp, ratio, translation_length, reference_length
    return bleu


def c_compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False, **kwargs):
    """Wrapper of ``fy_bleu.Scorer``, provide the same interface as Python implementation."""

    dict_ = kwargs.pop('dict', None)
    import fy_bleu
    scorer = fy_bleu.Scorer(dict_.pad_id, dict_.eos_id, dict_.unk_id)

    for references, translation in zip(reference_corpus, translation_corpus):
        for reference in references:
            scorer.add(reference, translation)

    return scorer.score(order=max_order) / 100.0


_compute_bleu_fn = None


def get_compute_bleu(mode=None):
    if mode is not None:
        if mode == 'c':
            return c_compute_bleu
        elif mode == 'py':
            return compute_bleu
        else:
            raise ValueError('Unknown mode {!r}'.format(mode))

    global _compute_bleu_fn
    if _compute_bleu_fn is None:
        try:
            from fy_bleu import Scorer
            _compute_bleu_fn = c_compute_bleu
        except ImportError:
            logging.warning('Package "fy_bleu" not installed. Use Python implementation to compute batch BLEU instead.')
            _compute_bleu_fn = compute_bleu
    return _compute_bleu_fn


def batch_bleu(generator, id_list, translation, ref_tokens, ref_dict):
    """Compute the BLEU of a batch.

    Args:
        generator:
        id_list:
        translation:
        ref_tokens: Tokenized reference tokens.
            list of sentences, sentences are list of (integer) tokens.
        ref_dict: Dictionary used to tokenize ref and sys sentences.

    Returns:

    """
    bleu_fn = get_compute_bleu()

    task = generator.task
    datasets = generator.datasets

    trans_str = (datasets.target_dict.string(t, bpe_symbol=task.BPESymbol, escape_unk=True) for t in translation)
    # trans_str = list(trans_str)
    # print('$', *trans_str, sep='\n')

    # [NOTE]: Translation and references must be tokenized with same dictionary.
    trans_corpus = [tokenizer.Tokenizer.tokenize(t, ref_dict, tensor_type=torch.IntTensor) for t in trans_str]

    ref_corpus = [[ref_tokens[i]] for i in id_list]
    # print('#', *[ref_dict.string(t[0], bpe_symbol=task.BPESymbol, escape_unk=True) for t in ref_corpus], sep='\n')

    result = bleu_fn(ref_corpus, trans_corpus, max_order=4, dict=ref_dict)

    # print('%', result)
    return result
