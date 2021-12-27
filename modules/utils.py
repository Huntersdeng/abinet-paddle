import cv2
import copy
import paddle
import numpy as np
import paddle.nn.functional as F

from paddle import nn, Tensor


def get_clones(module, N):
    return nn.LayerList([copy.deepcopy(module) for i in range(N)])

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def preprocess(img, width, height):
    img = np.array(img)
    h, w, _ = img.shape
    # if h>2*w:
    #     img = img.transpose(1,0,2)
    #     img = cv2.flip(img, 1)
    img = cv2.resize(img, (width, height))
    img = np.transpose(img, (2,0,1))
    scale = np.float32(1.0/255.0)
    mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1)).astype('float32')
    std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1)).astype('float32')

    return (img.astype('float32') * scale - mean) / std

def inv_preprocess(img):
    mean = paddle.to_tensor([0.485, 0.456, 0.406])
    std  = paddle.to_tensor([0.229, 0.224, 0.225])
    return img*std[...,None,None]+mean[...,None,None]


class CharsetMapper(object):
    """A simple class to map ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.
    """

    def __init__(self,
                 filename='',
                 max_length=30,
                 null_char=u'\u2591'):
        """Creates a lookup table.

        Args:
          filename: Path to charset file which maps characters to ids.
          max_sequence_length: The max length of ids and string.
          null_char: A unicode character used to replace '<null>' character.
            the default value is a light shade block '░'.
        """
        self.null_char = null_char
        self.max_length = max_length

        self.label_to_char = self._read_charset(filename)
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)
 
    def _read_charset(self, filename):
        """Reads a charset definition from a tab separated text file.

        Args:
          filename: a path to the charset file.

        Returns:
          a dictionary with keys equal to character codes and values - unicode
          characters.
        """
        import re
        pattern = re.compile(r'(\d+)\t(.+)')
        charset = {}
        self.null_label = 0
        charset[self.null_label] = self.null_char
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                m = pattern.match(line)
                assert m, f'Incorrect charset file. line #{i}: {line}'
                label = int(m.group(1)) + 1
                char = m.group(2)
                charset[label] = char
        return charset

    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, '')

    def get_text(self, labels, length=None, padding=True, trim=False):
        """ Returns a string corresponding to a sequence of character ids.
        """
        length = length if length else self.max_length
        labels = [l.item() if isinstance(l, Tensor) else int(l) for l in labels]
        if padding:
            labels = labels + [self.null_label] * (length-len(labels))
        text = ''.join([self.label_to_char[label] for label in labels])
        if trim: text = self.trim(text)
        return text

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """ Returns the labels of the corresponding text.
        """
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length

        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return '0123456789'

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
                valid_chars.append(c)
        return ''.join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)