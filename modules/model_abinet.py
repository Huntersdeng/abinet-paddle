import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .model import Model
from .model_vision import BaseVision
from .model_language import BCNLanguage
from .model_alignment import BaseAlignment

class ABINetModel(Model):
    def __init__(self, config):
        super().__init__(config['Global'])
        self.max_length = self.charset.max_length
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)

    def forward(self, images, *args):
        v_res = self.vision(images)
        v_tokens = F.softmax(v_res['logits'], axis=-1)
        v_lengths = v_res['pt_lengths'].clip(2, self.max_length)

        l_res = self.language(v_tokens, v_lengths)
        l_feature, v_feature = l_res['feature'], v_res['feature']
        
        a_res = self.alignment(l_feature, v_feature)
        return a_res, l_res, v_res

if __name__=='__main__':
    import yaml, paddle
    from PIL import Image
    from utils import preprocess
    device = 'gpu:0'
    device = paddle.set_device(device)
    with open('./configs/abinet.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = ABINetModel(cfg)
    model.eval()
    state = paddle.load('pretrain_models/pretrain_vm.pdparams')
    model.vision.load_dict(state)
    state = paddle.load('pretrain_models/pretrain_lm.pdparams')
    model.language.load_dict(state)
    state = paddle.load('pretrain_models/pretrain_alignment.pdparams')
    model.alignment.load_dict(state)

    img = Image.open('figs/football.jpg').convert('RGB')
    img = paddle.to_tensor(preprocess(img, 128 ,32)).unsqueeze(0)
    out, _, _ = model(img)
    pt_text, pt_scores, pt_lengths = model._get_text(out['logits'])
    print(pt_text, pt_scores, pt_lengths)