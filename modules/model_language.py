import paddle
import paddle.nn as nn

from .model import Model
from .transformer import (PositionalEncoding, 
                         TransformerDecoder,
                         TransformerDecoderLayer)

class BCNLanguage(Model):
    def __init__(self, config):
        super().__init__(config['Global'])

        cfg_language = config['Architecture']
        cfg_language['max_length'] = config['Global']['max_length']
        

        self.max_length = self.charset.max_length
        self.d_model = cfg_language['d_model']
        self.num_layers = cfg_language['Language']['num_layers']
        self.detach = cfg_language['Language']['detach']

        self.project = nn.Linear(self.charset.num_classes, self.d_model, bias_attr=False)
        self.token_encoder = PositionalEncoding(self.d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout=0, max_len=self.max_length)
        decoder_layer = TransformerDecoderLayer(**cfg_language)
        self.model = TransformerDecoder(decoder_layer, self.num_layers)

        self.cls = nn.Linear(self.d_model, self.charset.num_classes)

    def forward(self, tokens, lengths):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        if self.detach: tokens = tokens.detach()
        embed = self.project(tokens)  # (N, T, E)
        embed = embed.transpose((1, 0, 2))  # (T, N, E)
        embed = self.token_encoder(embed)  # (T, N, E)
        mask = self._get_mask(lengths, self.max_length)

        zeros = paddle.zeros_like(embed)
        qeury = self.pos_encoder(zeros)
        output = self.model(qeury, embed, memory_mask=mask)  # (T, N, E)
        output = output.transpose((1, 0, 2))  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res =  {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths}
        return res

if __name__=='__main__':
    import yaml, paddle
    import paddle.nn.functional as F
    with open('./configs/abinet.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = BCNLanguage(cfg)
    model.eval()
    state = paddle.load('pretrain_models/pretrain_lm.pdparams')
    model.load_dict(state)
    token = paddle.to_tensor([model.charset.get_labels('footfall'), model.charset.get_labels('bookxtore')])
    token = F.one_hot(token,37)
    length = paddle.to_tensor((9,10))
    out = model(token, length)
    pt_text, pt_scores, pt_lengths = model._get_text(out['logits'])
    print(pt_text, pt_scores)