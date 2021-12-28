import paddle
import paddle.nn as nn

from .resnet import resnet45
from .transformer import (PositionalEncoding,
                         TransformerEncoder,
                         TransformerEncoderLayer)


class ResTranformer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.resnet = resnet45()
        self.pos_encoder = PositionalEncoding(config['d_model'], max_len=8*32)
        encoder_layer = TransformerEncoderLayer(**config)
        self.transformer = TransformerEncoder(encoder_layer, config['num_layers'])

    def forward(self, images):
        feature = self.resnet(images)
        n, c, h, w = feature.shape
        feature = feature.reshape((n, c, -1)).transpose((2, 0, 1))
        feature = self.pos_encoder(feature)
        feature = self.transformer(feature)
        feature = feature.transpose((1, 2, 0)).reshape((n, c, h, w))
        return feature

if __name__=='__main__':
    import yaml
    with open('./configs/abinet.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)['Architecture']['Vision']
    backbone = ResTranformer(cfg['Backbone'])
    img = paddle.randn((2,3,32,128))
    x = backbone(img)
    print(x.shape)
