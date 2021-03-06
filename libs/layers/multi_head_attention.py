#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Multi-head attention layer."""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .base import ChildLayer, wrap_ppp
from .common import Linear
from ..utils import common
from .ppp import push_prepostprocessors

__author__ = 'fyabc'


def attention_and_proj_mask(
        layer, query, key, value, src_lengths,
        subsequent_mask=True, target_embedding=None, attn_mean=False, mask=None, time_first=False):
    """Wrap attention with input / output projection and mask computation.

    :math:`Attention(Q, K, V) = softmax( Q * K^T / \sqrt{d_head} ) * V`

    Args:
        layer:
        query (Tensor): (batch_size, length_q, d_model) of float32
            If time_first: (length_q, batch_size, d_model) of float32
        key (Tensor): (batch_size, length_kv, d_model) of float32
            If time_first: (length_kv, batch_size, d_model) of float32
        value (Tensor): (batch_size, length_kv, d_model) of float32
            If time_first: (length_kv, batch_size, d_model) of float32
        src_lengths (Tensor): (batch_size,) of float32
        subsequent_mask (bool):
        target_embedding (Tensor): (batch_size, length_q, d_model) of float32
            If time_first: (length_q, batch_size, d_model) of float32
        attn_mean (bool):
        mask (Tensor): (batch_size, 1, 1, d_model) of float32
            If time_first: (1, 1, batch_size, d_model) of float32
        time_first (bool):

    Returns:

    """

    qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
    kv_same = key.data_ptr() == value.data_ptr()

    h = layer.h
    d_head = layer.d_head

    # Mask: (batch_size, 1, src_seq_len)
    if mask is None:
        mask = common.pad_and_subsequent_mask(
            src_lengths, layer.in_encoder, apply_subsequent_mask=subsequent_mask, maxlen=key.size(1))
    batch_size = query.size(1) if time_first else query.size(0)

    # 1) Do all the linear projections in batch from d_model => h x d_head
    if qkv_same:
        q, k, v = layer.in_proj_qkv(query)
    elif kv_same:
        q = layer.in_proj_q(query)
        if key is None:
            assert value is None
            # this will allow us to concat it with previous value and get
            # just get the previous value   [NOTE]: Not implemented now
            k = v = q.new(0)
        else:
            k, v = layer.in_proj_kv(key)
    else:
        q = layer.in_proj_q(query)
        k = layer.in_proj_k(key)
        v = layer.in_proj_v(value)

    if target_embedding is not None:
        q = (q + target_embedding) * math.sqrt(0.5)
    q *= layer.scaling

    length_q = q.size(0) if time_first else q.size(1)
    length_kv = k.size(0) if time_first else k.size(1)

    # q: (batch_size, length_q, d_model)
    #   If time first: (length_q, batch_size, d_model)
    # k & v: (batch_size, length_kv, d_model)
    #   If time first: (length_kv, batch_size, d_model)

    # 2) Apply attention on all the projected vectors in batch.

    if time_first:
        q = q.contiguous().view(length_q, batch_size * h, d_head).transpose(0, 1)
        k = k.contiguous().view(length_kv, batch_size * h, d_head).transpose(0, 1)
        v = v.contiguous().view(length_kv, batch_size * h, d_head).transpose(0, 1)
    else:
        # Batch first.
        q = q.view(batch_size, -1, h, d_head).transpose(1, 2)
        k = k.view(batch_size, -1, h, d_head).transpose(1, 2)
        v = v.view(batch_size, -1, h, d_head).transpose(1, 2)

        q = q.contiguous().view(batch_size * h, length_q, d_head)
        k = k.contiguous().view(batch_size * h, length_kv, d_head)
        v = v.contiguous().view(batch_size * h, length_kv, d_head)

    # TODO: Simplify the code.
    r"""Compute scaled dot-product attention.

    :math:`Attention(Q, K, V) = softmax( Q * K^T / \sqrt{d_head} ) * V`

    Args:
        query (batch_size, num_heads, length_q, d_head):
        key (batch_size, num_heads, length_kv, d_head):
        value (batch_size, num_heads, length_kv, d_v):
        mask (batch_size, 1, 1 or length_q, length_kv):
        dropout:

    Returns:
        tuple
            Attention value (batch_size, num_heads, length_q, d_v):
            attn_weights (batch_size, num_heads, length_q, length_kv): Attention probability distribution
    """

    scores = th.bmm(q, k.transpose(1, 2))
    assert list(scores.size()) == [batch_size * h, length_q, length_kv]

    # [NOTE]: Skip if mask is None or mask is all 1 (no padding to mask)
    if mask is not None and not mask.all():
        # don't attend to padding symbols
        scores = scores.view(batch_size, h, length_q, length_kv)
        scores = scores.masked_fill_(mask == 0, float('-inf'))
        scores = scores.view(batch_size * h, length_q, length_kv)
    # FIXME: Attention score and QKV same.

    attn_weights = F.softmax(scores, dim=-1)
    dropout = layer.dropout
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    attn = th.bmm(attn_weights, v)
    assert list(attn.size()) == [batch_size * h, length_q, d_head]

    # x: Attention value (batch_size, num_heads, length_q, d_v)
    # attn: Attention probability distribution (batch_size, num_heads, length_q, length_kv)
    attn_weights = attn_weights.view(batch_size, h, length_q, length_kv)

    if attn_mean:
        attn_weights = attn_weights.mean(dim=1)

    # 3) "Concat" using a view and apply a final linear.
    # FIXME: Attention weights different.
    if time_first:
        attn = attn.transpose(0, 1).contiguous().view(length_q, batch_size, h * d_head)
    else:
        attn = attn.view(batch_size, h, length_q, d_head)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, h * d_head)

    attn = layer.out_proj(attn)

    return attn, attn_weights


class MultiHeadAttention(ChildLayer):
    r"""The module of multi-head attention.

    :math:`MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O`

    where :math:`head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)`

    where :math:`W_i^Q, W_i^K \in R^{d_{model} \times d_k}, W_i^V \in R^{d_{model} \times d_v}`

    and :math:`W^O \in R^{h * d_v \times d_{model}}`.

    Args:
        h: Number of heads.
        d_model: Model output size.
        dropout:
        window: Local attention window size, None means global attention.

    Inputs: query, key, value, mask
        - **query** (batch_size, length_q, d_model):
        - **key** (batch_size, length_kv, d_model):
        - **value** (batch_size, length_kv, d_model):
        - **mask** (batch_size, 1, 1 or length_q, length_kv) or None:

    Output:
        - **output** (batch_size, length_q, d_model):
    """

    def __init__(self, h, d_model, **kwargs):
        hparams = kwargs.pop('hparams', None)

        super().__init__(hparams)

        assert d_model % h == 0

        self.d_model = d_model
        self.d_head = d_model // h
        self.h = h
        self.d_q = kwargs.pop('d_q', d_model)
        self.d_kv = kwargs.pop('d_kv', d_model)     # [NOTE]: We assume that d_v always == d_k.
        self.scaling = self.d_head ** -0.5
        self.window = window = kwargs.pop('window', None)

        assert window is None or (isinstance(window, int) and window % 2 == 1), \
            'Local attention window size must be None or an odd number'

        # Input / output projections.
        linear_bias = kwargs.pop('linear_bias', True)
        self.dim_equal = self.d_q == self.d_kv == d_model

        if self.dim_equal:
            self.in_proj_weight = nn.Parameter(th.Tensor(3 * d_model, d_model))
            if linear_bias:
                self.in_proj_bias = nn.Parameter(th.Tensor(3 * d_model))
            else:
                self.register_parameter('in_proj_bias', None)
        else:
            self.linears = nn.ModuleList([
                Linear(self.d_q, d_model, hparams=hparams, bias=linear_bias),
                Linear(self.d_kv, d_model, hparams=hparams, bias=linear_bias),
                Linear(self.d_kv, d_model, hparams=hparams, bias=linear_bias),
            ])

        self.out_proj = nn.Linear(d_model, self.d_q, bias=linear_bias)

        self.attn = None
        self.dropout = nn.Dropout(p=kwargs.pop('dropout', 0.1))
        self.in_encoder = kwargs.pop('in_encoder', True)
        self.subsequent_mask = kwargs.pop('subsequent_mask', True)
        self.attn_mean = kwargs.pop('attn_mean', False)

        # PPP args: [pre-code, post-code]
        ppp_args = kwargs.pop('ppp_args', None)
        if ppp_args is not None:
            push_prepostprocessors(self, *ppp_args, [1, 1, self.d_q], [1, 1, self.d_q])

        self.reset_parameters()

    def reset_parameters(self):
        if self.dim_equal:
            nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.dim_equal and self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    @wrap_ppp(3)
    def forward(self, query, key, value, src_lengths, **kwargs):
        x, self.attn = attention_and_proj_mask(
            self, query, key, value, src_lengths=src_lengths, subsequent_mask=self.subsequent_mask,
            target_embedding=kwargs.pop('target_embedding', None), attn_mean=self.attn_mean,
            mask=kwargs.pop('mask', None), time_first=self.hparams.time_first,
        )
        return x

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.d_model).chunk(2, dim=-1)

    def in_proj_q(self, query):
        if self.dim_equal:
            return self._in_proj(query, end=self.d_model)
        else:
            return self.linears[0](query)

    def in_proj_k(self, key):
        if self.dim_equal:
            return self._in_proj(key, start=self.d_model, end=2 * self.d_model)
        else:
            return self.linears[1](key)

    def in_proj_v(self, value):
        if self.dim_equal:
            return self._in_proj(value, start=2 * self.d_model)
        else:
            return self.linears[2](value)

    def _in_proj(self, input_, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias

        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input_, weight, bias)

    # def out_proj(self, out):
    #     return self.linears[-1](out)

    def extra_repr(self):
        return '#heads={}, d_model={}, d_q={}, d_kv={}'.format(self.h, self.d_model, self.d_q, self.d_kv)


class PositionwiseFeedForward(ChildLayer):
    def __init__(self, d_model, d_ff, **kwargs):
        hparams = kwargs.pop('hparams', None)

        super().__init__(hparams)

        linear_bias = kwargs.pop('linear_bias', True)

        self.w_1 = Linear(d_model, d_ff, hparams=hparams, bias=linear_bias)
        self.w_2 = Linear(d_ff, d_model, hparams=hparams, bias=linear_bias)
        self.dropout = nn.Dropout(kwargs.pop('dropout', 0.1))

    @wrap_ppp
    def forward(self, x, **kwargs):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SelfAttention(ChildLayer):
    """Wraps multi-head attention into self attention.

    Inputs: x, lengths
        - **x** (batch_size, length, d_model):
        - **lengths** (batch_size,) or None:

    Output:
        - **output** (batch_size, length, d_model):
    """

    def __init__(self, hparams, h, d_model, d_ff, **kwargs):
        super().__init__(hparams)

        linear_bias = kwargs.pop('linear_bias', True)
        self.d_model = d_model

        self.in_encoder = kwargs.pop('in_encoder', True)
        # [NOTE]: Only apply subsequent mask in decoder.
        self.attention = MultiHeadAttention(
            h, d_model, dropout=kwargs.pop('dropout', self.hparams.attention_dropout),
            hparams=hparams, linear_bias=linear_bias, subsequent_mask=not self.in_encoder)

        # [NOTE]: If not in encoder, add enc-dec attention.
        if not self.in_encoder:
            self.encdec_attention = MultiHeadAttention(
                    h, d_model=hparams.trg_embedding_size, d_q=d_model, d_kv=hparams.src_embedding_size,
                    dropout=hparams.attention_dropout, in_encoder=False, hparams=hparams,
                    linear_bias=hparams.attn_linear_bias, subsequent_mask=False, attn_mean=True,
                    ppp_args=['', 'dan'],
                )
        else:
            self.encdec_attention = None

        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, dropout=kwargs.pop('ffn_dropout', self.hparams.ffn_dropout),
            hparams=hparams, linear_bias=linear_bias)

        # [NOTE]: The encoder-decoder attention layer may be inside this attention layer.
        # Used in decoder.
        # self.encdec_attention_layer_ref = None
        # self.encdec_attention_fwd = None
        self.attn_scores = None

    # def add_encdec_attention(self, layer, args=(), kwargs=None):
    #     if kwargs is None:
    #         kwargs = {}
    #     self.encdec_attention_layer_ref = ref(layer)
    #     self.encdec_attention_fwd = lambda x: layer(x, *args, **kwargs)

    def forward(self, x, lengths=None, **kwargs):
        """

        Args:
            x:
            lengths:
            **kwargs:

        Returns:

        Notes:
            The self-attention layer contains a multi-head attention layer and a position-wise feed-forward layer.
            Each need to be preprocessed and postprocessed.
        """

        # [NOTE]: Use pre-computed mask for self attention layer if available.
        attn_result = self.attention(x, x, x, src_lengths=lengths, mask=kwargs.get('mask', None))

        if self.encdec_attention is not None:
            encoder_out = kwargs['encoder_out']
            encdec_result = self.encdec_attention(
                attn_result, encoder_out['x'], encoder_out['y'],
                src_lengths=kwargs['src_lengths'],
                target_embedding=kwargs['target_embedding'],
                mask=encoder_out['src_mask'],
            )
            self.attn_scores = self.encdec_attention.attn
        else:
            encdec_result = attn_result

        # if self.encdec_attention_fwd is not None:
        #     encdec_result = self.encdec_attention_fwd(attn_result)
        #     self.attn_scores = self.encdec_attention_layer_ref().attn
        # else:
        #     encdec_result = attn_result

        result = self.feed_forward(encdec_result)

        return result

    def push_prepostprocessors(self, preprocess_code, postprocess_code, input_shape, output_shape):
        # [NOTE]: Different layer norm parameters between child layers.
        push_prepostprocessors(self.attention, preprocess_code, postprocess_code,
                               [1, 1, self.d_model], [1, 1, self.d_model])
        push_prepostprocessors(self.feed_forward, preprocess_code, postprocess_code,
                               [1, 1, self.d_model], [1, 1, self.d_model])
        # This layer does not have its own ppp.
        self.simplify()


class MultiHeadAttention2(nn.Module):
    """Multi-head attention, fairseq transformer version."""

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self._mask = None

        self.in_proj_weight = nn.Parameter(th.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(th.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        # TODO: Fix this code:
        # 1. Change T x B x C -> B x T x C
        # 2. Make interface compatible

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if saved_state is not None:
            if 'prev_key' in saved_state:
                k = th.cat((saved_state['prev_key'], k), dim=0)
            if 'prev_value' in saved_state:
                v = th.cat((saved_state['prev_value'], v), dim=0)
            saved_state['prev_key'] = k
            saved_state['prev_value'] = v
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = th.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if mask_future_timesteps and incremental_state is None:
            assert query.size() == key.size(), \
                'mask_future_timesteps only applies to self-attention'
            attn_weights += self.buffered_mask(attn_weights).unsqueeze(0)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = th.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :]
            if bias is not None:
                bias = bias[start:]
        return F.linear(input, weight, bias)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = th.triu(common.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = th.triu(common.fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return common.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        common.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )


__all__ = [
    'MultiHeadAttention',
    'PositionwiseFeedForward',
    'SelfAttention',
]
