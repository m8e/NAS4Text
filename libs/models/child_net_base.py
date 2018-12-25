#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..tasks import get_task
from ..layers.common import *
from ..layers.grad_multiply import GradMultiply
from ..utils import common
from ..utils.data_processing import LanguagePairDataset

__author__ = 'fyabc'


class ChildNetBase(nn.Module):
    _Subclasses = {}

    def __init__(self, net_code, hparams):
        super().__init__()
        self.net_code = net_code
        self.hparams = hparams
        self.task = get_task(hparams.task)

    @classmethod
    def register_child_net(cls, subclass):
        cls._Subclasses[subclass.__name__] = subclass
        return subclass

    @classmethod
    def get_net(cls, net_type):
        return cls._Subclasses[net_type]

    def forward(self, *args):
        raise NotImplementedError()

    def encode(self, src_tokens, src_lengths):
        raise NotImplementedError()

    def decode(self, encoder_out, src_lengths, trg_tokens, trg_lengths):
        raise NotImplementedError()

    def get_normalized_probs(self, net_output, log_probs=False):
        raise NotImplementedError()

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def max_encoder_positions(self):
        raise NotImplementedError()

    def max_decoder_positions(self):
        raise NotImplementedError()

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from state_dict into this module and
        its descendants.

        Overrides the method in nn.Module; compared with that method this
        additionally "upgrades" state_dicts from old checkpoints.
        """
        state_dict = self.upgrade_state_dict(state_dict)
        super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        raise NotImplementedError()

    def num_parameters(self):
        """Number of parameters."""
        return sum(p.data.numel() for p in self.parameters())


class EncDecChildNet(ChildNetBase):
    def __init__(self, net_code, hparams):
        super().__init__(net_code, hparams)
        self.encoder = None
        self.decoder = None

    def forward(self, src_tokens, src_lengths, trg_tokens, trg_lengths):
        """

        Args:
            src_tokens: (batch_size, src_seq_len) of int32
            src_lengths: (batch_size,) of long
            trg_tokens: (batch_size, trg_seq_len) of int32
            trg_lengths: (batch_size,) of long

        Returns:
            (batch_size, seq_len, tgt_vocab_size) of float32
        """
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out = self.decoder(encoder_out, src_lengths, trg_tokens, trg_lengths)

        return decoder_out

    def encode(self, src_tokens, src_lengths):
        return self.encoder(src_tokens, src_lengths)

    def decode(self, encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state=None):
        return self.decoder(encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state=incremental_state)

    def get_normalized_probs(self, net_output, log_probs=False):
        return self.decoder.get_normalized_probs(net_output, log_probs)

    def max_encoder_positions(self):
        return self.encoder.max_positions()

    def max_decoder_positions(self):
        return self.decoder.max_positions()

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions(), self.decoder.max_positions()

    def upgrade_state_dict(self, state_dict):
        state_dict = self.encoder.upgrade_state_dict(state_dict)
        state_dict = self.decoder.upgrade_state_dict(state_dict)
        return state_dict

    def _build_embed_tokens(self):
        hparams = self.hparams
        src_embed_tokens = Embedding(self.task.SourceVocabSize, hparams.src_embedding_size, self.task.PAD_ID,
                                     hparams=hparams)
        if hparams.share_src_trg_embedding:
            assert self.task.SourceVocabSize == self.task.TargetVocabSize, \
                'Shared source and target embedding weights implies same source and target vocabulary size, but got ' \
                '{}(src) vs {}(trg)'.format(self.task.SourceVocabSize, self.task.TargetVocabSize)
            assert hparams.src_embedding_size == hparams.trg_embedding_size, \
                'Shared source and target embedding weights implies same source and target embedding size, but got ' \
                '{}(src) vs {}(trg)'.format(hparams.src_embedding_size, hparams.trg_embedding_size)
            trg_embed_tokens = src_embed_tokens
        else:
            trg_embed_tokens = Embedding(self.task.TargetVocabSize, hparams.trg_embedding_size, self.task.PAD_ID,
                                         hparams=hparams)
        return src_embed_tokens, trg_embed_tokens


class ChildEncoderBase(nn.Module):
    def __init__(self, code, hparams, controller=None):
        super().__init__()
        self.code = code
        self.hparams = hparams
        self.task = get_task(hparams.task)
        self.controller = controller

    def _init_post(self, input_shape):
        controller = self.controller
        if controller is not None:
            s_encoder = controller.shared_weights.encoder
            self.out_norm = s_encoder.out_norm
            self.fc2 = s_encoder.fc2
            self.output_shape = s_encoder.output_shape
            return

        hparams = self.hparams

        if hparams.enc_out_norm:
            self.out_norm = LayerNorm(input_shape[2])
        else:
            self.out_norm = None

        if hparams.enc_output_fc or input_shape[2] != hparams.src_embedding_size:
            self.fc2 = Linear(input_shape[2], hparams.src_embedding_size, hparams=hparams)
        else:
            self.fc2 = None

        # Encoder output shape
        self.output_shape = th.Size([input_shape[0], input_shape[1], hparams.src_embedding_size])

    def _fwd_pre(self, src_tokens, src_lengths):
        x = src_tokens
        logging.debug('Encoder input shape: {}'.format(list(x.shape)))

        x = self.embed_tokens(x) * self.embed_scale + self.embed_positions(x)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)
        source_embedding = x

        # Compute mask from length, shared between all encoder layers.
        src_mask = self._mask_from_lengths(x, src_lengths, apply_subsequent_mask=False)

        # x = self.fc1(x)

        # x: (batch_size, src_seq_len, src_emb_size)
        # src_mask: (batch_size, 1 (broadcast to num_heads), 1, src_seq_len)
        # source_embedding: (batch_size, src_seq_len, src_emb_size)

        if self.hparams.time_first:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            source_embedding = source_embedding.transpose(0, 1)

        logging.debug('Encoder input shape after embedding: {}'.format(list(x.shape)))
        return x, src_mask, source_embedding

    def _fwd_post(self, x, src_mask, source_embedding):
        # Output normalization
        if self.out_norm is not None:
            x = self.out_norm(x)

        # project back to size of embedding
        if self.fc2 is not None:
            x = self.fc2(x)

        if self.hparams.apply_grad_mul:
            # scale gradients (this only affects backward, not forward)
            x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        if self.hparams.connect_src_emb:
            # add output to input embedding for attention
            y = (x + source_embedding) * math.sqrt(0.5)
        else:
            y = x

        logging.debug('Encoder output shape: {} & {}'.format(list(x.shape), list(y.shape)))
        return {
            'x': x,
            'y': y,
            'src_mask': src_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """Reorder encoder output according to new_order.

        Args:
            encoder_out:
            new_order:

        Returns:
            Reordered encoder out.
        """
        raise NotImplementedError()

    def _build_embedding(self, embed_tokens):
        controller = self.controller
        if controller is not None:
            s_encoder = controller.shared_weights.encoder
            self.embed_tokens = s_encoder.embed_tokens
            self.embed_positions = s_encoder.embed_positions
            self.embed_scale = s_encoder.embed_scale
            return

        hparams = self.hparams
        self.embed_tokens = embed_tokens
        self.embed_positions = PositionalEmbedding(
            hparams.max_src_positions,
            hparams.src_embedding_size,
            self.task.PAD_ID,
            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE,
            hparams=hparams,
            learned=hparams.enc_learned_pos,
        )
        self.embed_scale = math.sqrt(hparams.src_embedding_size) if hparams.embed_scale else 1

    def _mask_from_lengths(self, x, lengths, apply_subsequent_mask):
        return common.pad_and_subsequent_mask(
            lengths, in_encoder=True, apply_subsequent_mask=apply_subsequent_mask, maxlen=x.size(1))

    def max_positions(self):
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        return state_dict


class ChildDecoderBase(nn.Module):
    # A temporary flag to mark using incremental state or not.
    ApplyIncrementalState = False

    def __init__(self, code, hparams, controller=None):
        super().__init__()

        self.code = code
        self.hparams = hparams
        self.task = get_task(hparams.task)
        self.controller = controller

        self.embed_tokens = None
        self.embed_positions = None
        self.fc_last = None

    def _init_post(self, input_shape):
        controller = self.controller
        if controller is not None:
            s_decoder = controller.shared_weights.decoder
            self.output_shape = s_decoder.output_shape
            self.out_norm = s_decoder.out_norm
            self.fc2 = s_decoder.fc2
        else:
            hparams = self.hparams

            # Decoder output shape (before softmax)
            self.output_shape = input_shape

            if hparams.dec_out_norm:
                self.out_norm = LayerNorm(self.output_shape[2])
            else:
                self.out_norm = None
            if hparams.dec_output_fc or self.output_shape[2] != hparams.decoder_out_embedding_size:
                self.fc2 = Linear(self.output_shape[2], hparams.decoder_out_embedding_size, hparams=hparams)
            else:
                self.fc2 = None

        self._build_fc_last()

    def _build_embedding(self, embed_tokens):
        controller = self.controller
        if controller is not None:
            s_decoder = controller.shared_weights.decoder
            self.embed_tokens = s_decoder.embed_tokens
            self.embed_positions = s_decoder.embed_positions
            self.embed_scale = s_decoder.embed_scale
            return

        hparams = self.hparams

        self.embed_tokens = embed_tokens
        self.embed_positions = PositionalEmbedding(
            hparams.max_trg_positions,
            hparams.trg_embedding_size,
            self.task.PAD_ID,
            left_pad=LanguagePairDataset.LEFT_PAD_TARGET,
            hparams=hparams,
            learned=hparams.dec_learned_pos,
        )
        self.embed_scale = math.sqrt(hparams.trg_embedding_size) if hparams.embed_scale else 1

    def _build_fc_last(self):
        controller = self.controller
        if controller is not None:
            s_decoder = controller.shared_weights.decoder
            self.fc_last = s_decoder.fc_last
            return

        hparams = self.hparams

        if hparams.share_input_output_embedding:
            assert hparams.trg_embedding_size == hparams.decoder_out_embedding_size, \
                'Shared embed weights implies same dimensions out_embedding_size={} vs trg_embedding_size={}'.format(
                    hparams.decoder_out_embedding_size, hparams.trg_embedding_size)
            self.fc_last = None
        else:
            self.fc_last = nn.Parameter(th.Tensor(self.task.TargetVocabSize, hparams.decoder_out_embedding_size))
            nn.init.normal_(self.fc_last, mean=0, std=hparams.decoder_out_embedding_size ** -0.5)

    def _fwd_pre(self, encoder_out, src_lengths, trg_tokens, trg_lengths, incremental_state):
        # TODO: Implement incremental state.
        if not self.ApplyIncrementalState:
            incremental_state = None

        encoder_state_mean = self._get_encoder_state_mean(encoder_out, src_lengths)

        # split and (transpose) encoder outputs
        encoder_out = self._split_encoder_out(encoder_out, incremental_state)

        x = trg_tokens
        logging.debug('Decoder input shape: {}'.format(list(x.shape)))

        x = self._embed_tokens(x, incremental_state) * self.embed_scale + self.embed_positions(x, incremental_state)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)

        # Compute mask from length, shared between all decoder layers.
        trg_mask = self._mask_from_lengths(x, trg_lengths, apply_subsequent_mask=True)

        # x: (batch_size, trg_seq_len, src_emb_size)
        # trg_mask: (batch_size, 1 (broadcast to num_heads), trg_seq_len, src_seq_len)
        # target_embedding: (batch_size, src_seq_len, src_emb_size)

        if self.hparams.time_first:
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
        target_embedding = x

        logging.debug('Decoder input shape after embedding: {}'.format(list(x.shape)))
        return x, encoder_out, trg_mask, target_embedding, encoder_state_mean

    def _fwd_post(self, x, avg_attn_scores):
        if self.hparams.time_first:
            # T x B x C -> B x T x C
            x = x.transpose(0, 1)

        # Output normalization
        if self.out_norm is not None:
            x = self.out_norm(x)

        # Project back to size of vocabulary
        if self.fc2 is not None:
            x = self.fc2(x)
            x = F.dropout(x, p=self.hparams.dropout, training=self.training)

        if self.fc_last is None:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = F.linear(x, self.fc_last)

        logging.debug('Decoder output shape: {} & {}'.format(
            list(x.shape), None if avg_attn_scores is None else list(avg_attn_scores.shape)))
        return x, avg_attn_scores

    def _split_encoder_out(self, encoder_out, incremental_state):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = common.get_incremental_state(self, incremental_state, 'encoder_out')
        if cached_result is not None:
            return cached_result

        # transpose only once to speed up attention layers
        if self.hparams.enc_dec_attn_type == 'fairseq':
            # [NOTE]: Only do transpose here for fairseq attention
            encoder_a, encoder_b = encoder_out['x'], encoder_out['y']
            encoder_a = encoder_a.transpose(1, 2).contiguous()
            encoder_out['x'] = encoder_a
            encoder_out['y'] = encoder_b
        result = encoder_out

        if incremental_state is not None:
            common.set_incremental_state(self, incremental_state, 'encoder_out', result)
        return result

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # Keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _contains_lstm(self):
        """Test if the decoder contains LSTM layers."""
        return False

    def _get_encoder_state_mean(self, encoder_out, src_lengths):
        if not self._contains_lstm():
            return None

        enc_hidden = encoder_out['x']

        if self.hparams.time_first:
            enc_hidden = enc_hidden.transpose(0, 1)
        max_length = enc_hidden.size(1)
        src_mask = common.mask_from_lengths(src_lengths, left_pad=False, max_length=max_length, cuda=True)

        return (th.sum(enc_hidden * src_mask.unsqueeze(dim=2).type_as(enc_hidden), dim=1) /
                src_lengths.unsqueeze(dim=1).type_as(enc_hidden))

    @staticmethod
    def get_normalized_probs(net_output, log_probs=False):
        logits = net_output[0]
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def upgrade_state_dict(self, state_dict):
        return state_dict

    def max_positions(self):
        return self.embed_positions.max_positions()

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """

        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )

        self.apply(apply_reorder_incremental_state)

    def set_beam_size(self, beam_size):
        """Sets the beam size in the decoder and all children."""
        if getattr(self, '_beam_size', -1) != beam_size:
            def apply_set_beam_size(module):
                if module != self and hasattr(module, 'set_beam_size'):
                    module.set_beam_size(beam_size)

            self.apply(apply_set_beam_size)
            self._beam_size = beam_size

    def _mask_from_lengths(self, x, lengths, apply_subsequent_mask):
        return common.pad_and_subsequent_mask(
            lengths, in_encoder=False, apply_subsequent_mask=apply_subsequent_mask, maxlen=x.size(1))


class ChildIncrementalDecoderBase(ChildDecoderBase):
    """The incremental child decoder.

    TODO: Just a tag class now, move incremental-related methods into it in future."""
    pass


def forward_call(method_name):
    def _method(self, *args, **kwargs):
        return getattr(self.module, method_name)(*args, **kwargs)
    return _method


def forward_property(property_name):
    def _getter(self):
        return getattr(self.module, property_name)
    return property(fget=_getter, doc='The {!r} attribute of the module'.format(property_name))


class ParalleledChildNet(nn.DataParallel):
    encode = forward_call('encode')
    decode = forward_call('decode')
    encoder = forward_property('encoder')
    decoder = forward_property('decoder')
    get_normalized_probs = forward_call('get_normalized_probs')
    get_targets = forward_call('get_targets')
    max_encoder_positions = forward_call('max_encoder_positions')
    max_decoder_positions = forward_call('max_decoder_positions')
    max_positions = forward_call('max_positions')
    upgrade_state_dict = forward_call('upgrade_state_dict')
    num_parameters = forward_call('num_parameters')


__all__ = [
    'ChildNetBase',
    'EncDecChildNet',

    'ChildEncoderBase',
    'ChildDecoderBase',
    'ChildIncrementalDecoderBase',

    'forward_call',

    'ParalleledChildNet',
]
