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


def load_fairseq_checkpoint(model_path, model):
    """Load parameter weights from fairseq checkpoint (only for de-en e6d6 baseline now)

    Args:
        model_path (str):
        model:

    Returns:
        the updated model
    """

    state = th.load(model_path)
    fairseq_model = state['model']

    nas_model_dict = dict(model.named_parameters())

    def _assign(nas_key, fairseq_key):
        try:
            nas_model_dict[nas_key].data.copy_(fairseq_model[fairseq_key].data)
        except Exception:
            raise KeyError('Error when copy from {!r} to {!r}'.format(fairseq_key, nas_key))

    _assign('module.encoder.embed_tokens.weight', 'encoder.embed_tokens.weight')
    for i in range(6):
        _assign('module.encoder.layers.{}.nodes.2.postprocessors.0.weight'.format(i),
                'encoder.layers.{}.layer_norms.0.weight'.format(i))
        _assign('module.encoder.layers.{}.nodes.2.postprocessors.0.bias'.format(i),
                'encoder.layers.{}.layer_norms.0.bias'.format(i))
        _assign('module.encoder.layers.{}.nodes.2.op1.attention.in_proj_weight'.format(i),
                'encoder.layers.{}.self_attn.in_proj_weight'.format(i))
        _assign('module.encoder.layers.{}.nodes.2.op1.attention.in_proj_bias'.format(i),
                'encoder.layers.{}.self_attn.in_proj_bias'.format(i))
        _assign('module.encoder.layers.{}.nodes.2.op1.attention.out_proj.weight'.format(i),
                'encoder.layers.{}.self_attn.out_proj.weight'.format(i))
        _assign('module.encoder.layers.{}.nodes.2.op1.attention.out_proj.bias'.format(i),
                'encoder.layers.{}.self_attn.out_proj.bias'.format(i))
        _assign('module.encoder.layers.{}.nodes.3.postprocessors.0.weight'.format(i),
                'encoder.layers.{}.layer_norms.1.weight'.format(i))
        _assign('module.encoder.layers.{}.nodes.3.postprocessors.0.bias'.format(i),
                'encoder.layers.{}.layer_norms.1.bias'.format(i))
        _assign('module.encoder.layers.{}.nodes.3.op1.pffn.w_1.weight'.format(i),
                'encoder.layers.{}.fc1.weight'.format(i))
        _assign('module.encoder.layers.{}.nodes.3.op1.pffn.w_1.bias'.format(i),
                'encoder.layers.{}.fc1.bias'.format(i))
        _assign('module.encoder.layers.{}.nodes.3.op1.pffn.w_2.weight'.format(i),
                'encoder.layers.{}.fc2.weight'.format(i))
        _assign('module.encoder.layers.{}.nodes.3.op1.pffn.w_2.bias'.format(i),
                'encoder.layers.{}.fc2.bias'.format(i))
    _assign('module.decoder.embed_tokens.weight', 'decoder.embed_tokens.weight')
    for i in range(6):
        _assign('module.decoder.layers.{}.nodes.2.postprocessors.0.weight'.format(i),
                'decoder.layers.{}.self_attn_layer_norm.weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.2.postprocessors.0.bias'.format(i),
                'decoder.layers.{}.self_attn_layer_norm.bias'.format(i))
        _assign('module.decoder.layers.{}.nodes.2.op1.attention.in_proj_weight'.format(i),
                'decoder.layers.{}.self_attn.in_proj_weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.2.op1.attention.in_proj_bias'.format(i),
                'decoder.layers.{}.self_attn.in_proj_bias'.format(i))
        _assign('module.decoder.layers.{}.nodes.2.op1.attention.out_proj.weight'.format(i),
                'decoder.layers.{}.self_attn.out_proj.weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.2.op1.attention.out_proj.bias'.format(i),
                'decoder.layers.{}.self_attn.out_proj.bias'.format(i))
        _assign('module.decoder.layers.{}.nodes.3.postprocessors.0.weight'.format(i),
                'decoder.layers.{}.encoder_attn_layer_norm.weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.3.postprocessors.0.bias'.format(i),
                'decoder.layers.{}.encoder_attn_layer_norm.bias'.format(i))
        _assign('module.decoder.layers.{}.nodes.3.op1.attention.in_proj_weight'.format(i),
                'decoder.layers.{}.encoder_attn.in_proj_weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.3.op1.attention.in_proj_bias'.format(i),
                'decoder.layers.{}.encoder_attn.in_proj_bias'.format(i))
        _assign('module.decoder.layers.{}.nodes.3.op1.attention.out_proj.weight'.format(i),
                'decoder.layers.{}.encoder_attn.out_proj.weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.3.op1.attention.out_proj.bias'.format(i),
                'decoder.layers.{}.encoder_attn.out_proj.bias'.format(i))
        _assign('module.decoder.layers.{}.nodes.4.postprocessors.0.weight'.format(i),
                'decoder.layers.{}.final_layer_norm.weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.4.postprocessors.0.bias'.format(i),
                'decoder.layers.{}.final_layer_norm.bias'.format(i))
        _assign('module.decoder.layers.{}.nodes.4.op1.pffn.w_1.weight'.format(i),
                'decoder.layers.{}.fc1.weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.4.op1.pffn.w_1.bias'.format(i),
                'decoder.layers.{}.fc1.bias'.format(i))
        _assign('module.decoder.layers.{}.nodes.4.op1.pffn.w_2.weight'.format(i),
                'decoder.layers.{}.fc2.weight'.format(i))
        _assign('module.decoder.layers.{}.nodes.4.op1.pffn.w_2.bias'.format(i),
                'decoder.layers.{}.fc2.bias'.format(i))

    return model
