#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .child_net_base import EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase
from ..layers.nao_layer import NAOLayer
from ..layers.nas_controller import NASController
from ..layers.net_code import NetCode
from ..layers.common import *


class NAOChildEncoder(ChildEncoderBase):
    def __init__(self, hparams, embed_tokens):
        # [NOTE]: Does not use net code, pass ``None``.
        super().__init__(None, hparams)

        # Encoder input shape (after embedding).
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.src_embedding_size])

        # Embeddings.
        self._build_embedding(embed_tokens)

        # The main encoder network.
        self.layers = nn.ModuleList()
        input_shape = self.input_shape

        for i in range(hparams.num_encoder_layers):
            # [NOTE]: Shape not changed here.
            self.layers.append(NAOLayer(hparams, input_shape, in_encoder=True))

        self._init_post(input_shape)

    @property
    def num_layers(self):
        return len(self.layers)

    def reorder_encoder_out(self, encoder_out, new_order):
        raise RuntimeError('This method must not be called')

    def forward(self, *input):
        raise RuntimeError('This method must not be called')


class NAOChildDecoder(ChildIncrementalDecoderBase):
    def __init__(self, hparams, embed_tokens):
        # [NOTE]: Does not use net code, pass ``None``.
        super().__init__(None, hparams)

        # Decoder input shape (after embedding)
        # [NOTE]: The shape[0] (batch_size) and shape[1] (seq_length) is variable and useless.
        self.input_shape = th.Size([1, 1, hparams.trg_embedding_size])

        self._build_embedding(embed_tokens)

        # The main encoder network.
        self.layers = nn.ModuleList()
        input_shape = self.input_shape

        for i in range(hparams.num_decoder_layers):
            # [NOTE]: Shape not changed here.
            self.layers.append(NAOLayer(hparams, input_shape, in_encoder=False))

        self._init_post(input_shape)

    @property
    def num_layers(self):
        return len(self.layers)

    def forward(self, *input):
        raise RuntimeError('This method must not be called')


class NAOChildNet(EncDecChildNet):
    """The class of NAO child net.

    [NOTE]: This class is just a "container" of shared weights, the forward and backward methods will not be called.
    """
    def __init__(self, hparams):
        super().__init__(None, hparams)

        src_embed_tokens, trg_embed_tokens = self._build_embed_tokens()

        self.encoder = NAOChildEncoder(hparams, src_embed_tokens)
        self.decoder = NAOChildDecoder(hparams, trg_embed_tokens)


class NaoEpdAttention(nn.Module):
    def __init__(self, dim):
        super(NaoEpdAttention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = th.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = th.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = th.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class NaoEpd(nn.Module):
    InitRange = 0.04

    KeyAttnScore = 'attention_score'
    KeyLength = 'length'
    KeySequence = 'sequence'

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.enc_vocab_size = self.hparams.ctrl_enc_vocab_size
        self.dec_vocab_size = self.hparams.ctrl_dec_vocab_size
        self.num_enc_layers = self.hparams.ctrl_num_encoder_layers
        self.num_dec_layers = self.hparams.ctrl_num_decoder_layers
        self.num_mlp_layers = self.hparams.ctrl_num_mlp_layers
        self.enc_emb_size = self.hparams.ctrl_enc_emb_size
        self.enc_hidden_size = self.hparams.ctrl_enc_hidden_size
        self.mlp_hidden_size = self.hparams.ctrl_mlp_hidden_size
        self.dec_hidden_size = self.hparams.ctrl_dec_hidden_size
        self.enc_dropout_p = self.hparams.ctrl_enc_dropout
        self.mlp_dropout_p = self.hparams.ctrl_mlp_dropout
        self.dec_dropout_p = self.hparams.ctrl_dec_dropout
        self.src_length = self.hparams.ctrl_src_length
        self.enc_length = self.hparams.ctrl_enc_length
        self.dec_length = self.hparams.ctrl_dec_length
        self.eos_id = 0
        self.sos_id = 0

        self.encoder_emb = nn.Embedding(self.enc_vocab_size, self.enc_emb_size)
        self.encoder = nn.LSTM(
            self.enc_hidden_size,
            self.enc_hidden_size,
            self.num_enc_layers,
            batch_first=True,
            dropout=self.enc_dropout_p,
        )
        self.encoder_dropout = nn.Dropout(p=self.enc_dropout_p)

        p_weights = []
        for i in range(self.num_mlp_layers):
            if i == 0:
                size = self.enc_hidden_size, self.mlp_hidden_size
            else:
                size = self.mlp_hidden_size, self.enc_hidden_size
            p_weights.append(nn.Parameter(
                    th.Tensor(*size).uniform_(-self.InitRange, self.InitRange)))
        p_weights.append(nn.Parameter(
            th.Tensor(self.enc_hidden_size if self.num_mlp_layers == 0 else self.mlp_hidden_size, 1).uniform_(
                -self.InitRange, self.InitRange)))
        self.predictor = nn.ParameterList(p_weights)
        self.mlp_dropout = nn.Dropout(p=self.mlp_dropout_p)

        self.decoder_init_input = None
        self.decoder_emb = nn.Embedding(self.dec_vocab_size, self.dec_hidden_size)
        self.decoder = nn.LSTM(
            self.dec_hidden_size,
            self.dec_hidden_size,
            self.num_dec_layers,
            batch_first=True,
            dropout=self.dec_dropout_p,
        )
        self.decoder_attention = NaoEpdAttention(self.dec_hidden_size)
        self.decoder_dropout = nn.Dropout(self.dec_dropout_p)
        self.decoder_out = nn.Linear(self.dec_hidden_size, self.dec_vocab_size)

        self.decode_function = F.log_softmax

    def encode(self, encoder_input):
        embedded = self.encoder_emb(encoder_input)
        embedded = self.encoder_dropout(embedded)

        if self.src_length != self.enc_length:
            assert self.src_length % self.enc_length == 0
            ratio = self.src_length // self.enc_length
            embedded = embedded.view(-1, self.enc_length, ratio * self.enc_emb_size)
        out, hidden = self.encoder(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs, encoder_state = out, hidden

        out = th.mean(out, dim=1)
        out = F.normalize(out, 2, dim=1)
        arch_emb = out

        for i in range(self.num_mlp_layers):
            out = self.mlp_dropout(out)
            out = out.mm(self.W[i])
            out = F.relu(out)
        out = out.mm(self.W[-1])
        predict_value = F.sigmoid(out)

        return encoder_outputs, encoder_state, arch_emb, predict_value

    def encoder_infer(self, encoder_input, predict_lambda):
        encoder_outputs, encoder_state, arch_emb, predict_value = self.encode(encoder_input)
        grads_on_outputs = th.autograd.grad(predict_value, encoder_outputs, th.ones_like(predict_value))[0]
        new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb = th.mean(new_encoder_outputs, dim=1)
        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        return encoder_outputs, encoder_state, arch_emb, predict_value, new_encoder_outputs, new_arch_emb

    def decode_step(self, x, hidden, encoder_outputs, fn):
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.decoder_emb(x)
        embedded = self.decoder_dropout(embedded)
        output, hidden = self.decoder(embedded, hidden)
        output, attn = self.decoder_attention(output, encoder_outputs)

        predicted_softmax = fn(self.decoder_out(output.contiguous().view(-1, self.dec_hidden_size)), dim=1).view(
            batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def decode(self, x, encoder_hidden=None, encoder_outputs=None, fn=F.log_softmax):
        ret_dict = {
            self.KeyAttnScore: [],
        }

        x, batch_size, length = self._validate_decoder_args(x, encoder_hidden, encoder_outputs)
        assert length == self.dec_length
        decoder_hidden = self._init_decoder_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)

        def _seq2arch(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            ret_dict[self.KeyAttnScore].append(step_attn)
            # TODO: Change hard-coding here.
            if step % 2 == 0:  # sample index, should be in [1, step+1]
                symbols = decoder_outputs[-1][:, :step // 2 + 2].topk(1)[1]
            else:  # sample operation, should be in [12, 15]
                symbols = decoder_outputs[-1][:, 12:].topk(1)[1] + 12

            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            decoder_output, decoder_hidden, step_attn = self.decode_step(
                decoder_input, decoder_hidden, encoder_outputs, fn=fn)
            step_output = decoder_output.squeeze(1)
            symbols = _seq2arch(di, step_output, step_attn)
            decoder_input = symbols

        ret_dict[self.KeySequence] = sequence_symbols
        ret_dict[self.KeyLength] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def decoder_infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self.decode(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden

    # TODO: Change the format of output arch to be compatible with current code.

    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encode(input_variable)
        encoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decode(
            target_variable, encoder_hidden, encoder_outputs, fn=self.decode_function)
        decoder_outputs = th.stack(decoder_outputs, 0).permute(1, 0, 2)
        arch = th.stack(ret[self.KeySequence], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, arch

    def generate_new_arch(self, input_variable, predict_lambda=1):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = \
            self.encoder_infer(input_variable, predict_lambda)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decode(
            None, new_encoder_hidden, new_encoder_outputs, fn=self.decode_function)
        new_arch = th.stack(ret[self.KeySequence], 0).permute(1, 0, 2)
        return new_arch

    def _init_decoder_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden

    def _validate_decoder_args(self, x, encoder_hidden, encoder_outputs):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if x is None and encoder_hidden is None:
            batch_size = 1
        else:
            if x is not None:
                batch_size = x.size(0)
            else:
                batch_size = encoder_hidden[0].size(1)

        # set default input and max decoding length
        if x is None:
            x = th.autograd.Variable(th.LongTensor([self.sos_id] * batch_size).view(batch_size, 1))
            if th.cuda.is_available():
                x = x.cuda()
            max_length = self.decoder_length
        else:
            max_length = x.size(1)

        return x, batch_size, max_length

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()


class NAOController(NASController):
    SkipNAT = True  # FIXME: Flag. Skip NAT models or not.

    def __init__(self, hparams):
        super().__init__(hparams)

        # The model which contains shared weights.
        self.shared_weights = NAOChildNet(hparams)
        self._supported_ops_cache = {
            True: self._reversed_supported_ops(self.shared_weights.encoder.layers[0].supported_ops()),
            False: self._reversed_supported_ops(self.shared_weights.decoder.layers[0].supported_ops()),
        }

        # EPD.
        self.epd = NaoEpd(hparams)

    @staticmethod
    def _reversed_supported_ops(supported_ops):
        """Get reversed supported ops.

        Args:
            supported_ops (list):

        Returns:
            dict
        """
        return {
            (op_name, tuple(op_args)): i
            for i, (op_name, op_type, op_args) in enumerate(supported_ops)
        }

    def _codec(self, in_encoder):
        return self.shared_weights.encoder if in_encoder else self.shared_weights.decoder

    def _layer(self, in_encoder, layer_id):
        return self._codec(in_encoder).layers[layer_id]

    def get_weight(self, in_encoder, layer_id, index, input_index, op_code, **kwargs):
        # [NOTE]: ENAS sharing style, same as DARTS sharing style.
        op_args = kwargs.pop('op_args', [])
        op_idx = self._supported_ops_cache[in_encoder].get((op_code, tuple(op_args)), None)
        if op_idx is None:
            raise RuntimeError('The op type {} and op args {} does not exist in the controller'.format(
                op_code, op_args))
        layer = self._layer(in_encoder, layer_id)

        return layer.edges[layer.offsets[(input_index, index)]].ops[op_idx]

    def get_node_ppp(self, in_encoder, layer_id, index, **kwargs):
        layer = self._layer(in_encoder, layer_id)
        ppp = layer.node_ppp_list[index - layer.num_input_nodes]
        return {
            'pre': ppp.preprocessors,
            'post': ppp.postprocessors,
            'residual_projection': ppp.residual_projection,
        }

    def get_block_ppp(self, in_encoder, layer_id, **kwargs):
        layer = self._layer(in_encoder, layer_id)
        return {
            'pre': layer.preprocessors,
            'post': layer.postprocessors,
            'residual_projection': layer.residual_projection,
        }

    def cuda(self, device=None):
        self.shared_weights.cuda(device)
        self.epd.cuda(device)
        return self

    def _generate_block(self, layer: NAOLayer):
        result = []
        num_input_nodes = layer.num_input_nodes
        num_total_nodes = layer.num_total_nodes
        in_encoder = layer.in_encoder
        supported_ops = list(self._supported_ops_cache[in_encoder].keys())
        supported_ops_idx = list(range(len(supported_ops)))

        result.extend([[None for _ in range(2 * num_input_nodes + 1)] for _ in range(num_input_nodes)])

        while True:
            comp_nodes = []
            for j in range(num_input_nodes, num_total_nodes):
                edges = [np.random.randint(0, j) for _ in range(2)]
                ops = []
                for _ in range(2):
                    op_name, op_args = supported_ops[np.random.choice(supported_ops_idx)]
                    ops.append([op_name] + list(op_args))

                comp_nodes.append(
                    edges +
                    ops +
                    [layer.node_combine_op] +
                    layer.node_ppp_code
                )

            # [NOTE]: Skip invalid arches.
            if self._valid_arch(comp_nodes, in_encoder):
                break

        result.extend(comp_nodes)

        result.append({
            'preprocessors': layer.ppp_code[0],
            'postprocessors': layer.ppp_code[1],
        })

        return result

    def _template_net_code(self, e, d):
        return NetCode({
            'Type': 'BlockChildNet',
            'Global': {},
            'Blocks': {
                'enc1': e,
                'dec1': d,
            },
            'Layers': [
                ['enc1' for _ in range(self.shared_weights.encoder.num_layers)],
                ['dec1' for _ in range(self.shared_weights.decoder.num_layers)],
            ]
        })

    def _valid_arch(self, comp_nodes, in_encoder):
        # 1. If #layers >= 2, must contains an input index 0, or the first layer will be disconnected.
        ed = self.shared_weights.encoder if in_encoder else self.shared_weights.decoder
        num_layers = len(ed.layers)
        if num_layers > 1:
            if all(n[0] != 0 and n[1] != 0 for n in comp_nodes):
                return False

        # 2. Decoder must contains at least one "EncoderAttention" layer.
        if not in_encoder:
            if all(n[2][0] != 'EncoderAttention' and n[3][0] != 'EncoderAttention' for n in comp_nodes):
                return False

        # 3. Decoder must contains
        if self.SkipNAT and not in_encoder:
            nat_layers = 'SelfAttention', 'CNN', 'LSTM'
            if all(n[2][0] not in nat_layers and n[3][0] not in nat_layers for n in comp_nodes):
                return False

        return True

    def generate_arch(self, n):
        enc0 = self.shared_weights.encoder.layers[0]
        dec0 = self.shared_weights.decoder.layers[0]

        return [self._template_net_code(self._generate_block(enc0), self._generate_block(dec0)) for _ in range(n)]

    def parse_arch_to_seq(self, arch):
        """Parse architecture to sequence.

        Format:
            seq of enc1 + seq of dec1
            -> seq of block: [seq of ops]
            -> seq of op: [in1, op1_index, in2, op2_index]
            -> inX: Integer in [0, num_total_nodes - 1]
            -> opX_index: Integer in [num_total_nodes, num_total_nodes + num_total_ops - 1]

            num_total_nodes = hparams.num_total_nodes
            For each inX of node[i], inX in [0, i - 1]

        Args:
            arch (NetCode):

        Returns:
            A list of integers.

        # TODO: Add doctest here.
        """

        num_total_nodes = self._layer(True, 0).num_total_nodes

        def _parse_block(block, in_encoder):
            _so = self._supported_ops_cache[in_encoder]
            seq = []
            for node in block:
                in1, in2, op1, op2, *_ = node
                for in_, op in zip((in1, in2), (op1, op2)):
                    op_name, *op_args = op
                    op_idx = _so[op_name, tuple(op_args)]
                    seq.extend([in_, op_idx + num_total_nodes])

        return _parse_block(arch.blocks['enc1'], True) + _parse_block(arch.blocks['dec1'], False)

    def predict(self, topk_arches):
        pass

    def get_arch_parameter_size(self, arch, exclude_emb=True):
        """Get parameter size of the given architecture."""
        # TODO
