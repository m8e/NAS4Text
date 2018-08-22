#! /usr/bin/python
# -*- coding: utf-8 -*-

import collections
import logging
import math

from . import tokenizer, dictionary
from ..tasks import get_task

__author__ = 'fyabc'


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
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.

    Copied from TensorFlow <https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py>.

    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
        precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
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

    return bleu, precisions, bp, ratio, translation_length, reference_length


def fy_compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Wrapper of ``fy_bleu.Scorer``, provide the same interface as Python implementation."""
    pass


_compute_bleu_fn = None


def get_compute_bleu(mode=None):
    if mode is not None:
        if mode == 'c':
            return fy_compute_bleu
        elif mode == 'py':
            return compute_bleu
        else:
            raise ValueError('Unknown mode {!r}'.format(mode))

    global _compute_bleu_fn
    if _compute_bleu_fn is None:
        try:
            import fy_bleu
            _compute_bleu_fn = fy_compute_bleu
        except ImportError:
            logging.warning('Package "fy_bleu" not installed. Use Python implementation to compute batch BLEU instead.')
            _compute_bleu_fn = compute_bleu
    return _compute_bleu_fn


def batch_bleu(generator, sample, translation):
    """Compute the BLEU of a batch."""
    bleu_fn = get_compute_bleu()

    task = generator.task
    datasets = generator.datasets

    dict_ = dictionary.Dictionary(None, task=task, mode='empty')

    # TODO: Read ref str from dataset
    trans_str = datasets.trg_dict.string(translation, bpe_symbol=task.BPESymbol, escape_unk=True)

    print(sample.keys(), sample['net_input'].keys(), sample['id'])

    return 0.0
