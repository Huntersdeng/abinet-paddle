import paddle
import math
import paddle.nn.functional as F
from paddle.nn import (Dropout, LayerNorm, Linear, 
                      Layer, Sequential, Upsample,
                      Conv2D, ReLU, BatchNorm2D)
from utils import get_activation_fn, get_clones


class TransformerEncoder(Layer):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = paddle.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Layer):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        d_inner: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = paddle.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, d_inner=2048, dropout=0.1, 
                 activation="relu", **kwargs):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, d_inner)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(d_inner, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2, attn = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class TransformerDecoder(Layer):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, memory_mask=None):
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, memory_mask=memory_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerDecoderLayer(Layer):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        d_inner: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, d_inner=2048, dropout=0.1, 
                 activation="relu", **kwargs):
        super(TransformerDecoderLayer, self).__init__()
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
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, memory_mask=None):
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # if self.has_self_attn:
        #     tgt2, attn = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
        #                         key_padding_mask=tgt_key_padding_mask)
        #     tgt = tgt + self.dropout1(tgt2)
        #     tgt = self.norm1(tgt)
        tgt2, attn2 = self.multihead_attn(tgt, memory, memory, mask=memory_mask)

        # if self.siamese:
        #     tgt3, attn3 = self.multihead_attn2(tgt, memory2, memory2, attn_mask=memory_mask2,
        #                     key_padding_mask=memory_key_padding_mask2)
        #     tgt = tgt + self.dropout2(tgt3)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt

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

def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return Sequential(Conv2D(in_c, out_c, k, s, p),
                         BatchNorm2D(out_c),
                         ReLU(True))

def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return Sequential(Upsample(size=size, scale_factor=scale_factor, 
                                     mode=mode, align_corners=align_corners),
                         Conv2D(in_c, out_c, k, s, p),
                         BatchNorm2D(out_c),
                         ReLU(True))


class PositionAttention(Layer):
    def __init__(self, config):
        super().__init__()
        max_length = config['max_length'] + 1
        self.max_length = max_length
        in_channels = config['in_channels']
        num_channels = config['num_channels']
        w,h = config['output_size']
        mode = config['mode']
        self.k_encoder = Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder = Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(h, w), mode=mode)
        )

        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = Linear(in_channels, in_channels)

    def forward(self, x):
        N, E, H, W = x.shape
        k, v = x, x  # (N, E, H, W)

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        # TODO q=f(q,k)
        zeros = x.new_zeros((self.max_length, N, E))  # (T, N, E)
        q = self.pos_encoder(zeros)  # (T, N, E)
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)
        
        # calculate attention
        attn_scores = paddle.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = paddle.softmax(attn_scores, axis=-1)

        v = v.permute(0, 2, 3, 1).reshape((N, -1, E))  # (N, (H*W), E)
        attn_vecs = paddle.bmm(attn_scores, v)  # (N, T, E)

        return attn_vecs, attn_scores.reshape((N, -1, H, W))