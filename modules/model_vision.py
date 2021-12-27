import logging

from paddle.nn import Linear

from model import Model
from backbone import ResTranformer
from attention import PositionAttention


class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config['Global'])
        cfg_vision = config['Architecture']
        cfg_vision['max_length'] = config['Global']['max_length']
        cfg_vision['num_layers'] = cfg_vision['Vision']['num_layers']
        self.backbone = ResTranformer(cfg_vision)
        self.head = PositionAttention(**cfg_vision)
        self.cls = Linear(cfg_vision['d_model'], self.charset.num_classes)

    def forward(self, images):
        features = self.backbone(images)
        attn_vecs, attn_scores = self.head(features)
        logits = self.cls(attn_vecs) # (N, T, C)
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores}

if __name__=='__main__':
    import yaml, paddle
    from PIL import Image
    from utils import preprocess
    device = 'gpu:0'
    device = paddle.set_device(device)
    with open('./configs/abinet.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = BaseVision(cfg)
    model.eval()
    state = paddle.load('pretrain_models/pretrain_vm.pdparams')
    model.load_dict(state)

    img = Image.open('figs/CANDY.png').convert('RGB')
    img = paddle.to_tensor(preprocess(img, 128 ,32)).unsqueeze(0)
    out = model(img)
    pt_text, pt_scores, pt_lengths = model._get_text(out['logits'])
    print(pt_text, pt_scores, pt_lengths)