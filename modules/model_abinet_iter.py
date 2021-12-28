import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .model import Model
from .model_vision import BaseVision
from .model_language import BCNLanguage
from .model_alignment import BaseAlignment

class ABINetIterModel(Model):
    def __init__(self, config):
        super().__init__(config['Global'])
        self.iter_size = config['Architecture']['iter_size']
        self.max_length = self.charset.max_length
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)

    def forward(self, images, *args):
        v_res = self.vision(images)
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            tokens = F.softmax(a_res['logits'], axis=-1)
            lengths = a_res['pt_lengths'].clip(2, self.max_length)
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
        if self.training:
            return all_a_res, all_l_res, v_res
        else:
            return a_res, all_l_res[-1], v_res