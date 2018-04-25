#! /usr/bin/python
# -*- coding: utf-8 -*-

import torch as th

__author__ = 'fyabc'


_CurrentState = None
_Total = 0
_Correct = 0
_Loss = 0.0


def get_valid_first_token_likelihood(model, targets, net_output):
    global _Total, _Correct, _CurrentState, _Loss

    if model.training != _CurrentState:
        _Total, _Correct, _Loss = 0, 0, 0.0
        print('Refresh to mode {}'.format(model.training))
    _CurrentState = model.training

    probs_0 = model.get_normalized_probs(net_output, log_probs=False)[:, 0]
    targets_0 = targets[:, 0]

    max_probs, tokens_0 = th.max(probs_0, dim=-1)

    equals = (targets_0 == tokens_0).data

    _Total += equals.numel()
    _Correct += equals.sum()
    _Loss += -th.log(max_probs.data).sum()

    print('Total: {}; Correct: {}; Accuracy: {}; Loss {}'.format(_Total, _Correct, _Correct / _Total, _Loss / _Total))


def get_first_token_accuracy(ref_filename, translated_filename):
    with open(ref_filename, 'r', encoding='utf-8') as f_ref, open(translated_filename, 'r', encoding='utf-8') as f_trans:
        n_total = 0
        n_correct = 0
        for ref, trans in zip(f_ref, f_trans):
            ref0 = ref.split()[0]
            trans0 = trans.split()[0]

            n_total += 1
            if ref0 == trans0:
                n_correct += 1

        print('Total: {}; Correct: {}; Accuracy: {}'.format(n_total, n_correct, n_correct / n_total))
