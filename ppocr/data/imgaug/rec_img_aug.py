import math
import cv2
import numpy as np
import random
from PIL import Image
from .text_image_aug import tia_perspective, tia_stretch, tia_distort


class ABIRecResizeImage(object):
    def __init__(self, image_shape, scale=None, mean=None, std=None, **kwargs):
        self.image_shape = image_shape
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, data):
        img = data['image']
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        height, width = self.image_shape
        img = cv2.resize(img, (width, height))
        img = np.transpose(img, (2,0,1))
        data['image'] = (
            img.astype('float32') * self.scale - self.mean) / self.std
        return data
