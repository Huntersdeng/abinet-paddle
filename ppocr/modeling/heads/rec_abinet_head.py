from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


from paddle.nn import Dropout, LayerNorm, Linear, Layer

def get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

class ABIHead(Layer):
    def __init__(self, in_channels, out_channels, max_length, num_layers, detach, iter_size, nhead, d_inner=2048, dropout=0.1, 
                 activation="relu", norm=None, **kwargs):
        super().__init__()

        self.iter_size = iter_size
        self.max_length = max_length + 1

        self.cls = nn.Linear(in_channels, out_channels)
        self.language = BCNLanguage(in_channels, out_channels, self.max_length, num_layers, detach, nhead, d_inner=d_inner, dropout=dropout, 
                 activation=activation, norm=norm)
        self.alignment = Alignment(in_channels, out_channels)
        

    def forward(self, feature, targets=None):
        v_res = {'feature': feature, 'logits': self.cls(feature)}
        v_res['pt_lengths'] = self._get_length(v_res['logits'])
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = F.softmax(a_res['logits'], axis=-1)
            lengths = a_res['pt_lengths'].clip(2, self.max_length)
            feature, logits = self.language(tokens, lengths)
            l_res = {'feature': feature, 'logits': logits, 'pt_lengths': self._get_length(logits)}
            all_l_res.append(l_res)
            logits = self.alignment(l_res['feature'], v_res['feature'])
            a_res = {'logits': logits, 'pt_lengths': self._get_length(logits)}
            all_a_res.append(a_res)
        if self.training:
            return all_a_res, all_l_res, v_res
        else:
            return a_res, all_l_res[-1], v_res

    def _get_length(self, logit, axis=-1):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(axis=-1) == 0)
        abn = out.any(axis)
        out = ((out.astype(paddle.int32).cumsum(axis) == 1) & out).astype(paddle.int32)
        out = out.argmax(axis, dtype='int32')
        out = out + 1  # additional end token
        out = paddle.where(abn, out, paddle.to_tensor(logit.shape[1], dtype=paddle.int32).broadcast_to(out.shape))
        return out


class BCNLanguage(Layer):
    __constants__ = ['norm']

    def __init__(self, in_channels, out_channels, max_length, num_layers, detach, nhead, d_inner=2048, dropout=0.1, 
                 activation="relu", norm=None, **kwargs):
        super(BCNLanguage, self).__init__()
        self.num_layers = num_layers
        self.detach = detach
        self.norm = norm
        self.max_length = max_length

        self.project = nn.Linear(out_channels, in_channels, bias_attr=False)
        self.token_encoder = PositionalEncoding(in_channels, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=self.max_length)
        layer = BCNLayer(in_channels, nhead, d_inner=d_inner, dropout=dropout, activation=activation)
        self.layers = get_clones(layer, num_layers)

        self.cls = nn.Linear(in_channels, out_channels)

    def forward(self, tokens, lengths):
        if self.detach: tokens = tokens.detach()
        embed = self.project(tokens)  # (N, T, E)
        embed = embed.transpose((1, 0, 2))  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        mask = self._get_mask(lengths, self.max_length)

        zeros = paddle.zeros_like(embed)
        qeury = self.pos_encoder(zeros)
        output = self.layers_forward(qeury, embed, memory_mask=mask)  # (T, N, E)
        feature = output.transpose((1, 0, 2))  # (N, T, E)

        logits = self.cls(feature)  # (N, T, C)
        return feature, logits

    def layers_forward(self, tgt, memory, memory_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, memory_mask=memory_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    @staticmethod
    def _get_mask(lengths, max_length):
        masks = []
        for length in lengths:
            N = int(length)
            location_mask = -paddle.eye(max_length) > -1
            padding_mask = paddle.zeros((max_length,max_length))
            padding_mask[:N,:N] = 1
            padding_mask = padding_mask > 0
            mask = location_mask & padding_mask
            mask = (paddle.cast(mask, paddle.float32) - 1.0) * 1e9
            masks.append(mask)
        masks = paddle.stack(masks)
        return masks

class BCNLayer(Layer):

    def __init__(self, d_model, nhead, d_inner=2048, dropout=0.1, 
                 activation="relu"):
        super(BCNLayer, self).__init__()
        # if self.has_self_attn:
        #     self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        #     self.norm1 = LayerNorm(d_model)
        #     self.dropout1 = Dropout(dropout)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, d_inner)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_inner, d_model)

        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(BCNLayer, self).__setstate__(state)

    def forward(self, tgt, memory, memory_mask=None):
        tgt2, attn2 = self.multihead_attn(tgt, memory, memory, mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Alignment(Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(2 * in_channels, in_channels)
        self.cls = nn.Linear(in_channels, out_channels)

    def forward(self, l_feature, v_feature):
        """
        Args:
            l_feature: (N, T, E) where T is length, N is batch size and d is dim of model
            v_feature: (N, T, E) shape the same as l_feature 
            l_lengths: (N,)
            v_lengths: (N,)
        """
        f = paddle.concat((l_feature, v_feature), axis=2)
        f_att = F.sigmoid(self.linear(f))
        output = f_att * v_feature + (1 - f_att) * l_feature

        logits = self.cls(output)  # (N, T, C)

        return logits


class PositionalEncoding(Layer):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = paddle.zeros((max_len, d_model))
        position = paddle.arange(0, max_len, dtype=paddle.float32).unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, d_model, 2).astype(paddle.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = paddle.sin(position * div_term)
        pe[:, 1::2] = paddle.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose([1,0,2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class MultiHeadAttention(Layer):
    def __init__(self, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = get_clones(Linear(d_model, d_model), 4)
        self.dropout = Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        tgt_len, nbatches, embed_size = query.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).reshape((tgt_len, nbatches*self.h, self.d_k)).transpose((1,0,2))
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = self.attention(query, key, value, mask=mask,
                                     dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose((1,0,2)) \
            .reshape((tgt_len, nbatches, embed_size))

        return self.linears[-1](x), attention_map.sum(axis=1) / self.h


    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        nbatches = query.shape[0]//self.h
        tgt_len = query.shape[1]
        d_k = query.shape[2]

        scores = paddle.matmul(query, key.transpose((0,2,1))) \
                / math.sqrt(d_k)
        scores = scores.reshape((nbatches, self.h, tgt_len, -1))
        if mask is not None:
            # print(mask)
            scores = scores + mask
        else:
            pass

        p_attn = F.softmax(scores, axis=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return paddle.matmul(p_attn.reshape((nbatches*self.h, tgt_len, -1)), value), p_attn