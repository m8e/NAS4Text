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

    NoEOS = True    # FIXME: Flag. The decoder does not output <EOS>.

    def __init__(self, hparams, controller: 'NAOController'):
        super().__init__()
        self.hparams = hparams

        self.enc_vocab_size = controller.expected_vocab_size(True) if self.hparams.ctrl_enc_vocab_size is None \
            else self.hparams.ctrl_enc_vocab_size
        self.dec_vocab_size = controller.expected_vocab_size(False) if self.hparams.ctrl_dec_vocab_size is None \
            else self.hparams.ctrl_dec_vocab_size
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

        src_length = controller.expected_source_length()
        self.src_length = src_length
        self.enc_length = src_length if self.hparams.ctrl_enc_length is None else self.hparams.ctrl_enc_length
        self.dec_length = src_length if self.hparams.ctrl_dec_length is None else self.hparams.ctrl_dec_length
        self.eos_id, self.sos_id = 0, 0
        self.index_range = controller.expected_index_range()
        self.enc_op_range, self.dec_op_range = controller.expected_op_range(True), controller.expected_op_range(False)
        self.num_input_nodes = controller.example_layer().num_input_nodes

        self.encoder_emb = nn.Embedding(self.enc_vocab_size, self.enc_emb_size)
        self.encoder = nn.LSTM(
            self.enc_emb_size,
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

        self.logging_self()

    def logging_self(self):
        logging.info('Controller model structure:\n{}'.format(self))
        logging.info('All model parameters:')
        num_parameters = 0
        for name, param in self.named_parameters():
            logging.info('\t {}: {}, {}'.format(name, list(param.shape), param.numel()))
            num_parameters += param.numel()
        logging.info('Number of parameters: {}'.format(num_parameters))

    def encode(self, encoder_input):
        embedded = self.encoder_emb(encoder_input)
        embedded = self.encoder_dropout(embedded)

        # Embedded shape: (batch_size, src_length, src_emb_size)

        if self.src_length != self.enc_length:
            assert self.src_length % self.enc_length == 0
            ratio = self.src_length // self.enc_length
            embedded = embedded.view(-1, self.enc_length, ratio * self.enc_emb_size)

        # TODO: CuDNN error here on Azure017: "indexSelectLargeIndex" assertion failed and CUDNN_STATUS_NOT_INITIALIZED.
        out, hidden = self.encoder(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs, encoder_state = out, hidden

        out = th.mean(out, dim=1)
        out = F.normalize(out, 2, dim=1)
        arch_emb = out

        for i in range(self.num_mlp_layers):
            out = self.mlp_dropout(out)
            out = out.mm(self.predictor[i])
            out = F.relu(out)
        out = out.mm(self.predictor[-1])
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
        """

        Args:
            x:
            encoder_hidden:
            encoder_outputs:
            fn:

        Returns:

        """
        ret_dict = {
            self.KeyAttnScore: [],
        }

        x, batch_size, length = self._validate_decoder_args(x, encoder_hidden, encoder_outputs)
        assert length == self.dec_length
        decoder_hidden = self._init_decoder_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)

        def _repr2seq(step, step_output, step_attn):
            """Sample the sequence from the decoder output representation.

            Args:
                step (int):
                step_output:
                step_attn:

            Returns:
                Tensor (batch_size, 1)
            """
            decoder_outputs.append(step_output)
            ret_dict[self.KeyAttnScore].append(step_attn)

            # Split the step into (in_encoder, node_index, i).
            # i: [in1, op1, in2, op2]
            ed, ed_idx = divmod(step, length // 2)
            in_encoder = ed == 0
            node_idx, i = divmod(ed_idx, 4)

            op_range = self.enc_op_range if in_encoder else self.dec_op_range

            if i in (0, 2):     # Input index, should be in [1, node_idx + num_input_nodes + 1)
                if self.NoEOS:
                    symbols = step_output[:, 1:node_idx + self.num_input_nodes + 1].argmax(dim=-1).unsqueeze(1) + 1
                else:
                    # TODO: If allow EOS, need to generate EOS at i == 0.
                    raise NotImplementedError('EOS in decoder is not implemented now.')
            else:   # i in (1, 3), Op index, should be in [num_total_nodes, num_total_nodes + num_ops)
                symbols = step_output[:, op_range[0]:op_range[1]].argmax(dim=-1).unsqueeze(1) + op_range[0]
            sequence_symbols.append(symbols)

            if not self.NoEOS:
                # If generate EOS, check it and modify lengths.
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
            symbols = _repr2seq(di, step_output, step_attn)
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
        """

        Args:
            input_variable:
            predict_lambda:

        Returns:
            Tensor (batch_size, source_length, 1)
        """
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
            max_length = self.dec_length
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
        self._supported_ops_r_cache = {
            True: self._reversed_supported_ops(self.shared_weights.encoder.layers[0].supported_ops()),
            False: self._reversed_supported_ops(self.shared_weights.decoder.layers[0].supported_ops()),
        }
        self._supported_ops_cache = {
            True: {v: k for k, v in self._supported_ops_r_cache[True].items()},
            False: {v: k for k, v in self._supported_ops_r_cache[False].items()},
        }

        # EPD.
        self.epd = NaoEpd(hparams, self)

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
        op_idx = self._supported_ops_r_cache[in_encoder].get((op_code, tuple(op_args)), None)
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

    def _gen_input_nodes(self, layer: NAOLayer):
        num_input_nodes = layer.num_input_nodes
        return [[None for _ in range(2 * num_input_nodes + 1)] for _ in range(num_input_nodes)]

    def _gen_node_ppp(self, layer: NAOLayer):
        return {
            'preprocessors': layer.ppp_code[0],
            'postprocessors': layer.ppp_code[1],
        }

    def _generate_block(self, layer: NAOLayer):
        result = []
        num_input_nodes = layer.num_input_nodes
        num_total_nodes = layer.num_total_nodes
        in_encoder = layer.in_encoder
        supported_ops = self._supported_ops_cache[in_encoder]
        supported_ops_idx = list(range(len(supported_ops)))

        result.extend(self._gen_input_nodes(layer))

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
            if self.valid_arch(comp_nodes, in_encoder):
                break

        result.extend(comp_nodes)

        result.append(self._gen_node_ppp(layer))

        return result

    def template_net_code(self, e, d):
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

    def valid_arch(self, block, in_encoder):
        # Remove input nodes and block ppp.
        block = [n for n in block if isinstance(n, list) and n[0] is not None]

        # 1. If #layers >= 2, must contains an input index 0, or the first layer will be disconnected.
        ed = self.shared_weights.encoder if in_encoder else self.shared_weights.decoder
        num_layers = len(ed.layers)
        if num_layers > 1:
            if all(n[0] != 0 and n[1] != 0 for n in block):
                return False

        # 2. Decoder must contains at least one "EncoderAttention" layer.
        if not in_encoder:
            if all(n[2][0] != 'EncoderAttention' and n[3][0] != 'EncoderAttention' for n in block):
                return False

        # 3. Decoder must contains
        if self.SkipNAT and not in_encoder:
            nat_layers = 'SelfAttention', 'CNN', 'LSTM'
            if all(n[2][0] not in nat_layers and n[3][0] not in nat_layers for n in block):
                return False

        return True

    def generate_arch(self, n):
        enc0 = self.shared_weights.encoder.layers[0]
        dec0 = self.shared_weights.decoder.layers[0]

        return [self.template_net_code(self._generate_block(enc0), self._generate_block(dec0)) for _ in range(n)]

    # Arch - Sequence transforming and related methods.

    def parse_arch_to_seq(self, arch):
        """Parse architecture to sequence.

        Format:
            seq of enc1 + seq of dec1
            -> seq of block: [seq of ops]
            -> seq of op: [in1, op1_index, in2, op2_index]
            -> inX: Integer in [1, num_total_nodes - 1]
                (real value + 1)
            -> opX_index: Integer in [num_total_nodes, num_total_nodes + num_total_ops - 1]
                (real value + num_total_nodes)

            num_total_nodes = hparams.num_total_nodes
            For each inX of node[i], inX in [1, i]

        Args:
            arch (NetCode):

        Returns:
            A list of integers.
            Length: 2 (enc/dec) * #nodes * 4 (2 inputs + 2 ops)

        # TODO: Add doctest here.
        """

        num_total_nodes = self._layer(True, 0).num_total_nodes

        def _parse_block(block, in_encoder):
            _so = self._supported_ops_r_cache[in_encoder]
            seq = []
            for node in block:
                if not isinstance(node, list):
                    continue
                in1, in2, op1, op2, *_ = node
                if in1 is None:
                    continue
                for in_, op in zip((in1, in2), (op1, op2)):
                    op_name, *op_args = op
                    op_idx = _so[op_name, tuple(op_args)]
                    seq.extend([in_ + 1, op_idx + num_total_nodes])
            return seq

        return _parse_block(arch.blocks['enc1'], True) + _parse_block(arch.blocks['dec1'], False)

    def parse_seq_to_blocks(self, seq):
        length = self.expected_source_length()
        assert len(seq) == length, 'The length is expected to be {}, but got {}'.format(length, len(seq))

        num_total_nodes = self._layer(True, 0).num_total_nodes

        enc1, dec1 = [], []

        enc0 = self.shared_weights.encoder.layers[0]
        dec0 = self.shared_weights.decoder.layers[0]

        enc1.extend(self._gen_input_nodes(enc0))
        dec1.extend(self._gen_input_nodes(dec0))

        ins = []
        ops = []

        for index, x in enumerate(seq):
            ed, ed_idx = divmod(index, length // 2)
            in_encoder = True if ed == 0 else False
            node, i = divmod(ed_idx, 4)

            block = enc1 if in_encoder else dec1
            layer = enc0 if in_encoder else dec0
            _supported_ops = self._supported_ops_cache[in_encoder]

            if i == 0:
                # Refresh.
                ins, ops = [], []

            if i in (0, 2):
                # Add input index.
                ins.append(x - 1)
            if i in (1, 3):
                op_name, op_args = _supported_ops[x - num_total_nodes]
                ops.append([op_name] + list(op_args))

            if i == 3:
                # Append the new node.
                block.append(ins + ops + [layer.node_combine_op] + layer.node_ppp_code)

        enc1.append(self._gen_node_ppp(enc0))
        dec1.append(self._gen_node_ppp(dec0))

        return enc1, dec1

    def expected_source_length(self):
        """Get the expected source length of the sequence.
        See ``NAOController.parse_arch_to_seq`` for the details of the equation.
        """
        return 2 * self.hparams.num_nodes * 4

    def expected_index_range(self):
        """Get the [low, high) range of the input indices."""
        return 1, self._layer(True, 0).num_total_nodes

    def expected_op_range(self, in_encoder):
        """Get the [low, high) range of the op indices."""
        num_total_nodes = self._layer(True, 0).num_total_nodes
        return num_total_nodes, num_total_nodes + len(self._supported_ops_r_cache[in_encoder])

    def expected_vocab_size(self, in_encoder):
        """Get the expected vocabulary size."""
        return self.expected_op_range(in_encoder)[1]

    def example_layer(self, in_encoder=True):
        return self._layer(in_encoder, 0)

    def predict(self, topk_arches):
        pass

    def get_arch_parameter_size(self, arch, exclude_emb=True):
        """Get parameter size of the given architecture."""
        # TODO
