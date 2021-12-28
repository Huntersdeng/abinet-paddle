import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .model import Model

class BaseAlignment(Model):
    def __init__(self, config):
        super().__init__(config['Global'])
        d_model = config['Architecture']['d_model']
        self.max_length = self.charset.max_length  # additional stop token
        self.linear = nn.Linear(2 * d_model, d_model)
        self.cls = nn.Linear(d_model, self.charset.num_classes)

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
        pt_lengths = self._get_length(logits)

        return {'logits': logits, 'pt_lengths': pt_lengths}