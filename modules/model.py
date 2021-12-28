import paddle
import paddle.nn as nn

from paddle.nn import functional as F

from .utils import CharsetMapper


class Model(nn.Layer):

    def __init__(self, config):
        super().__init__()
        self.max_length = config['max_length'] + 1
        self.charset = CharsetMapper(config['character_dict_path'], max_length=self.max_length)
    
    def load(self, source, device=None, strict=True):
        state = paddle.load(source, map_location=device)
        self.load_state_dict(state['model'], strict=strict)

    def _get_length(self, logit, axis=-1):
        """ Greed decoder to obtain length from logit"""
        out = (logit.argmax(axis=-1) == self.charset.null_label)
        abn = out.any(axis)
        out = ((out.astype(paddle.int32).cumsum(axis) == 1) & out).astype(paddle.int32)
        out = out.argmax(axis, dtype='int32')
        out = out + 1  # additional end token
        out = paddle.where(abn, out, paddle.to_tensor(logit.shape[1], dtype=paddle.int32).broadcast_to(out.shape))
        return out

    def _get_text(self, logit):
        """ Greed decode """
        out = F.softmax(logit, axis=2)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = self.charset.get_text(o.argmax(axis=1), padding=False, trim=False)
            text = text.split(self.charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(axis=1))
            pt_lengths.append(min(len(text) + 1, self.charset.max_length))  # one for end-token
        return pt_text, pt_scores, pt_lengths

    # @staticmethod
    # def _get_padding_mask(length, max_length):
    #     length = length.unsqueeze(-1)
    #     grid = paddle.arange(0, max_length).unsqueeze(0)
    #     return grid >= length

    # @staticmethod
    # def _get_square_subsequent_mask(sz, diagonal=0, fw=True):
    #     r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    #         Unmasked positions are filled with float(0.0).
    #     """
    #     mask = (paddle.triu(paddle.ones(sz, sz), diagonal=diagonal) == 1)
    #     if fw: mask = mask.transpose((1, 0))
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

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
