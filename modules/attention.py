import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from .transformer import PositionalEncoding

class Attention(nn.Layer):
    def __init__(self, in_channels=512, max_length=25, n_feature=256):
        super().__init__()
        self.max_length = max_length

        self.f0_embedding = nn.Embedding(max_length, in_channels)
        self.w0 = nn.Linear(max_length, n_feature)
        self.wv = nn.Linear(in_channels, in_channels)
        self.we = nn.Linear(in_channels, max_length)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(axis=2)

    def forward(self, enc_output):
        enc_output = enc_output.transpose((0, 2, 3, 1)).flatten(1, 2)
        reading_order = paddle.arange(self.max_length, dtype=paddle.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.shape[0], -1)  # (S,) -> (B, S)
        reading_order_embed = self.f0_embedding(reading_order)  # b,25,512

        t = self.w0(reading_order_embed.transpose((0, 2, 1)))  # b,512,256
        t = self.active(t.transpose((0, 2, 1)) + self.wv(enc_output))  # b,256,512

        attn = self.we(t)  # b,256,25
        attn = self.softmax(attn.transpose((0, 2, 1)))  # b,25,256
        g_output = paddle.bmm(attn, enc_output)  # b,25,512
        return g_output, attn.reshape((*attn.shape[:2], 8, 32))


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2D(in_c, out_c, k, s, p),
                         nn.BatchNorm2D(out_c),
                         nn.ReLU(True))

def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = False if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor, 
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2D(in_c, out_c, k, s, p),
                         nn.BatchNorm2D(out_c),
                         nn.ReLU(True))


class PositionAttention(nn.Layer):
    def __init__(self, max_length, in_channels=512, num_channels=64, 
                 h=8, w=32, mode='nearest', **kwargs):
        super().__init__()
        self.max_length = max_length + 1
        self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(h, w), mode=mode)
        )

        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=self.max_length)
        self.project = nn.Linear(in_channels, in_channels)

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
        zeros = paddle.zeros((self.max_length, N, E))  # (T, N, E)
        q = self.pos_encoder(zeros)  # (T, N, E)
        q = q.transpose((1, 0, 2))  # (N, T, E)
        q = self.project(q)  # (N, T, E)
        
        # calculate attention
        attn_scores = paddle.bmm(q, k.flatten(2, 3))  # (N, T, (H*W))
        attn_scores = attn_scores / (E ** 0.5)
        attn_scores = F.softmax(attn_scores, axis=-1)

        v = v.transpose((0, 2, 3, 1)).reshape((N, -1, E))  # (N, (H*W), E)
        attn_vecs = paddle.bmm(attn_scores, v)  # (N, T, E)

        return attn_vecs, attn_scores.reshape((N, -1, H, W))

if __name__=='__main__':
    model = PositionAttention(25)
    img = paddle.randn((2,512,8,32))
    attn_vecs, attn_scores = model(img)
    print(attn_vecs.shape, attn_scores.shape)