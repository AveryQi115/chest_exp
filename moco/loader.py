# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter, ImageOps,Image
import random
import cv2
import numpy as np


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class EqualizeHist(object):

    def __call__(self, x):
        x = ImageOps.equalize(x)
        return x

class Convert2RGB(object):
    
    def __call__(self,x):
        x = x.convert('RGB')
        return x

